# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-681f87e7efb93a7fe2c0ca8693ed89cbd98e19a9b1cd66b4274d0a49023027a7`
- fixture hash: `sha256-a3933e3ff4510f68b788c54f4766a4413b4c8e5f41767b34e144aa18224f9ea0`
- score hash: `sha256-ed66518531f824dd4169b14c669d5acace9d1a599b99885c07c6ef39c6a816ba`
- bundle hash: `sha256-70310113c02e0c19d779b826e64e76a90cc4bd7fb257bbe4a3b8b99acc3904a6`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d3fa9eac509428c290df212041800a61f9388237d673568a94abb436cef2cd1a |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-4cd2711532ddf07d05cd8cc864824a6cf0f33dd28f856fa3848e343d7f663893 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-1986434bbb7ee38393594f294cf80b0aa79b4434fff4f169b069107506866882 |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-71398bb3ad5b0bddc9eb564d35b44fcafa9ae1bec5d82bb1697471de98d72724 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-544ec7c2 | sha256-29f535d1d4867c24edb2ecf878c01124915158b1f8e53a2c137cb64525a96a02 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-544ec7c2 | sha256-044b756a9e0b4b91f99eea77c79a100c0974366bdd685c32366c04f11d8d6623 |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-544ec7c2 | sha256-29f535d1d4867c24edb2ecf878c01124915158b1f8e53a2c137cb64525a96a02 |
