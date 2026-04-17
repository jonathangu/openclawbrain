# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-db3086ba9270f5f47434d4a3c708e73ff2624adb056b71992e75ebf839a91592`
- fixture hash: `sha256-2a8d321cab2bd435ac998d63d68b17b7fce95e9a0ea6d02ef75e09676d4240bd`
- score hash: `sha256-110d2f24cb573db5c720dac309dec8f87b3786e5221e4f2964b05c735b8040f7`
- bundle hash: `sha256-cd2145a5118af6cf5260929b3770c76d4a5afad9cca68af82bb73ff2bd7c5a71`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-da6633515ce74e28e9f8bbc2cc587b6b0548deffeab6470c77c67fc675828106 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-729bb7abf05f5ada16a17aa2e683fe99eac5a86f6e394fb1b827ff6ff84267ae |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-e71b86930a9e07a01b3f13627d020cfffd95e3e3c7a53c080f6dd228a4ee66e9 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-aa5400f4f8af393c66bc79302dee430098391c682c9c32306b44884a70320e1b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-33dfa328 | sha256-bb66ae7f02467e140de1b6dd608331125736b7a397bd690f1f46bf8f5f682423 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-33dfa328 | sha256-e842b87854f30db50e75d6cebba11fb966bc6d00ec8dfd9e54f80518f8702160 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-51038ec5 | sha256-1d0e6fd45c402b2fef6ca0afb16d5742a054f09e682dc6b58d9824762c259d26 |
