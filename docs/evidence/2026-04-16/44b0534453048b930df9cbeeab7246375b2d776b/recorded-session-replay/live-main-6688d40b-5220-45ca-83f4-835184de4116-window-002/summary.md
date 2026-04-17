# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2ed84d12aa80219c71f67ac4b4dc49c9c31220d644ee2203f557cfbb2718f653`
- fixture hash: `sha256-a5d98f20c022a45dcdc79196fa677af12fe3ae7a1d81ee01512e8a79553eb0a0`
- score hash: `sha256-0eb12253298b27a8b2731f3d072e7034b846d8239f8efb9cc652cec0b6df7d8e`
- bundle hash: `sha256-fc548a79b1260945b80a0d00d8be27b990207c120686a891f883b9867cdf919f`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2d0401b09442b39c74248dfb10f1b77d9b52939def1349d2c685ecac4f520b39 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b7738ce1c4a6a254019518d0be878c64b78bea843a04c83ae41cf2c64416ffcc |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8dc027f6f9ef485847f5fdbde6b811561bec19f98c2d969096418d62c5c2585f |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-ca10a79486d4aa7891433a725913662a7b6813dce09bdff621276e6d238bb7f5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f91f5fbb | sha256-aa1cd391add8080a40fbee4bdbd45cd0c25de9e2c9a2b7f6416b65fec231622a |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f91f5fbb | sha256-ea41647529cfd65eacd1cd68a9ecec38e55e0eab3ddcb0dae9d708d5a4a848a5 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-0d93d422 | sha256-0c981cdbda23228d6d7716f2ae9337fcb5323c9feb1fcaaae1b12b3ebfe66060 |
