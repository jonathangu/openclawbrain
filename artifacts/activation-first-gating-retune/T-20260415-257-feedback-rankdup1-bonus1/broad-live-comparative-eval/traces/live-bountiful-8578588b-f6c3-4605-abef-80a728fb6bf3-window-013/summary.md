# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-013`
- winner mode: `graph_prior_only`
- trace hash: `sha256-02e7d65886ec7c9c662814c419da7cda053558fd638bbd6332be92bb767d2ff2`
- fixture hash: `sha256-6508603acac99ba814ef6cb1f3424ef1d7247c3d69d4612af9ca33edb2806300`
- score hash: `sha256-7db955fe0c2ce8b1c73324b157591d37041425f10dcafe925a39f04da8198ca9`
- bundle hash: `sha256-16f83ac98401aff54067dc3b11e3723bca685561c6c194a7e14ffc55b6a91a3b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-39c1106c4dbd631708b383c103b666734b15b0322fe3241c472e8fc7fee74258 |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-964435d9d4a7613dea3c8a3cc4da2edc29332af9482d42d7dbc13f873846ea00 |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-97076d6515dbe63cd6afe12bdb1129c087f452a959de29415691bcf6097b4235 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-44126bdc580a13ecc0646ec21ccc3adbcbdc326b5aa9713fd9a46fa5dfb2b018 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-2b18278b | sha256-651f28a78b88dac4a0ae44b00cf4d74dec5a9e703960b7557c0cb69842d20533 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-2b18278b | sha256-4db3e3490e73718cb38a205e3f30dc9ed1a02aa4dfc12ec1267b7c5e6a310ead |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-2b18278b | sha256-651f28a78b88dac4a0ae44b00cf4d74dec5a9e703960b7557c0cb69842d20533 |
