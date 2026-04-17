# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-076`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6f8a152004f1762e8eed1ecc9dffc7029171fd911a8be7a9ecd27602349fd8ea`
- fixture hash: `sha256-fef2416147c50461e059554c89ffc13514d9838c25717ac0bb496af10eda074b`
- score hash: `sha256-e5616dd297b35357f5d1f81aba4fbde24970c0bbef57291c1217bcbc2bc66d6f`
- bundle hash: `sha256-73e414855757c04eb555d313543e4a19fb70fbc88eb7f91158bc3a00d69d3aaf`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-510726e0c2b1103191bca21eb76122edbdd44953bed132c7c6febf953ec52703 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fec9ec20db2aca4ee84e6cac32341574bd1417b3ebcd67bb6d680e6c76101ecc |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8ac1895ebf33525bb934178116a09f991785881fc1e5e121948cee17ce0fcdae |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-c08c952de0abf07bf45ab3f7fe4a348c9b994ff0ac46c2db6fedb5d6cbbc4067 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5b74a50c | sha256-cca50179b7898787b3e41786a2482620d62d5604fcd23155a12fc18a4e639d0b |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5b74a50c | sha256-32504f692e808d5338739245989f3c8c52e0357a75153701911f5e3d3993f38e |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-fb259629 | sha256-e10c3ad5097bfde00bc6cf8d77267c03299d8f96309b222d5e7946a146589379 |
