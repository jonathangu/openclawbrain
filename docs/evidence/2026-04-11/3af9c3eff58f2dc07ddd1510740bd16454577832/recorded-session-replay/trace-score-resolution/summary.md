# Recorded Session Replay Proof Bundle

- trace id: `trace-score-resolution`
- winner mode: `learned_route`
- trace hash: `sha256-574a64bf53c3d6173d64b044723abb88b3517de393d177eab13af71fafd23432`
- fixture hash: `sha256-63d53f199a24fc52c99e70ab08c081d9f795c9234d4c9ad6b641f3f9480003ab`
- score hash: `sha256-70e1e890ae54d5440109b8ef75576c5cefd731b5cc96717ed21dbfc872989afc`
- bundle hash: `sha256-59b200ffa3484dc22b6db1783284b1feba0ef67da8ee3de7b251e6a51585ee7a`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | learned_route | 100 |
| 2 | graph_prior_only | 70 |
| 3 | vector_only | 70 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 9/12
- compile ok rate: 0.75
- phrase hits: 8/16
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 3 | 0 | 0 | 0 | 1 |
| vector_only | 3 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 3 | 1 | 0.5 | 0 | 1 |
| learned_route | 3 | 1 | 1 | 0.666667 | 1 |

## Hardening Snapshot
- compile failures: 3/12
- compile failure rate: 0.25
- warnings: 0
- promotions: 2

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 3 | 0 | 3 | 3 |
| vector_only | 0 | 0 | 0 | 3 | 3 |
| graph_prior_only | 0 | 0 | 0 | 3 | 3 |
| learned_route | 0 | 0 | 2 | 3 | 3 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 3 | 0 | 0/4 | 0 | 0 | 3 | 2 | 0 | sha256-271076f35fa438fafb2771d3e4fdf49b2bf41b0468ccbbb99a0d1f5bee4f354a |
| vector_only | 3 | 3 | 2/4 | 0 | 0 | 3 | 2 | 0 | sha256-de0af0c6c158705dce3c57a7a377bbff134993e4a05da7b52d5892ef33b8fd2e |
| graph_prior_only | 3 | 3 | 2/4 | 0 | 0 | 3 | 2 | 0 | sha256-3b44f59b869b531eb431a9ff3808c4d78d4f9df6c56ceafdc365b21a659fa2bc |
| learned_route | 3 | 3 | 4/4 | 2 | 2 | 3 | 2 | 0 | sha256-54b29a21dbdff16d8e8753a3ca5d1889ef711ba43aa853b6be93492291203216 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | plan-turn-1 | 0 | no | 0/1 | no | no | none | none |
| no_brain | plan-turn-2 | 0 | no | 0/1 | no | no | none | none |
| no_brain | plan-turn-3 | 0 | no | 0/2 | no | no | none | none |
| vector_only | plan-turn-1 | 100 | yes | 1/1 | no | no | pack-bb756721 | sha256-13434e6346d0befe3d35b172f0e6751d575a1b2f10ecfca33069a08cf6b54412 |
| vector_only | plan-turn-2 | 100 | yes | 1/1 | no | no | pack-bb756721 | sha256-256e6af6cac4ea209c5aa2a66bfdd19568291544a6e472e304482fea68aea3f4 |
| vector_only | plan-turn-3 | 40 | yes | 0/2 | no | no | pack-bb756721 | sha256-13434e6346d0befe3d35b172f0e6751d575a1b2f10ecfca33069a08cf6b54412 |
| graph_prior_only | plan-turn-1 | 100 | yes | 1/1 | no | no | pack-bb756721 | sha256-13434e6346d0befe3d35b172f0e6751d575a1b2f10ecfca33069a08cf6b54412 |
| graph_prior_only | plan-turn-2 | 100 | yes | 1/1 | no | no | pack-bb756721 | sha256-256e6af6cac4ea209c5aa2a66bfdd19568291544a6e472e304482fea68aea3f4 |
| graph_prior_only | plan-turn-3 | 40 | yes | 0/2 | no | no | pack-bb756721 | sha256-13434e6346d0befe3d35b172f0e6751d575a1b2f10ecfca33069a08cf6b54412 |
| learned_route | plan-turn-1 | 100 | yes | 1/1 | no | yes | pack-bb756721 | sha256-13434e6346d0befe3d35b172f0e6751d575a1b2f10ecfca33069a08cf6b54412 |
| learned_route | plan-turn-2 | 100 | yes | 1/1 | yes | yes | pack-20bd38a1 | sha256-33d65676e03e8aa239b6aed891cf3939b6056a51f01ee19d98f033b197ad3e50 |
| learned_route | plan-turn-3 | 100 | yes | 2/2 | yes | no | pack-e2ebeddc | sha256-10a42a93de65533202b18e287be25c31a83120355520ed11aa889e9be391a132 |
