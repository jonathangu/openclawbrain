# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-044`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ea42e44c1f4e55e54482460d49f94e95de2212323681b79ab4ddfefd7592f32d`
- fixture hash: `sha256-7145b0c661dfe4e1efd2da3fdd776c9b7910c0b1be56f04f06cb4a5ee20cf473`
- score hash: `sha256-a11163a6f80ac710f1e68283bd7523610c61d52140a2b2bfa5add18887880422`
- bundle hash: `sha256-86c945f1a8169a985305eeb45d853d4c898c538761e48591f0e1ec3305b75cd8`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4c76d7fd1d12ceb5bfd4785d2783131a1a6cdf46c225ec13d0667a11e9c25468 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-79f153caa59c3ba58b58a5fd27113b2d8475f67e3ccb80b11e224dbf3b669513 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6ac0cb3b37402ad1a031b89a887f2380b2dfd23e067a27c3253579eac6741a97 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-e1c18088713fb28293204ecb165215ff831c306230248bb39a8499406a43630b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-aafc94d2 | sha256-ed2cd0d8963ed6d1e1319f683f6b1f0c8070af15f11e2598428289a828c72ec4 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-aafc94d2 | sha256-2147d31b9f7669a10cbc1016e829664ea8af24188fc2bc306d326aea4492d638 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-aafc94d2 | sha256-ed2cd0d8963ed6d1e1319f683f6b1f0c8070af15f11e2598428289a828c72ec4 |
