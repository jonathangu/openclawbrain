# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-019`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3d68f21a3db07e083abd55cb8f30309dffa35aea63874e95510f19d0d69cb1ce`
- fixture hash: `sha256-370af296b8752ce6655fe59921b05e957209333f8adae37b056699cf10a9af35`
- score hash: `sha256-84cfed3ecc1984c56acbd0c975ff9cc2b55b6cfc3d9f61c66d7c30c44546cef9`
- bundle hash: `sha256-e4aac8b9b664a5bbe4a81a67c937e48b0023dd91acca92d2b183be26a052d4b4`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b13ac363ce7285d3640914d39071894fd6c80687f14f6807f8531ccb47249088 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-836f9d8d924c9b75bd2d75d5a93747f925d268cdafd45c94b20c28aa97c2d47d |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-2ef3b27d8e315dac1881c4c925f96cb198ce94ca973172f20b12b77f8d476642 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-9a183d562aac03a6a17d2901f51b29299de155d281e2e9ecb47f8dfac2c2d443 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-98348118 | sha256-e5e14b6a5c9a63797bf764e0b1c82004785a8ec2ce097094dd038bc619fc13a0 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-98348118 | sha256-0d0a93d7ec1198debcb9c4ca75648ad89661f750a6851ec9042dccdea2ebb469 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-a65248a3 | sha256-d5bce59b4999c97c5e65470faf1777c4db4a9e3cabd9ad1d10d3fa729e14154a |
