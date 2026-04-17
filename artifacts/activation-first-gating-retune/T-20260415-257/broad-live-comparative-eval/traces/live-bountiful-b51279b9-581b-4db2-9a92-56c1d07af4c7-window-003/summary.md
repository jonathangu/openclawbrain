# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b6443732e15180ade7a5127a9fabf70925f6863db8ba4387cc6be009639e6099`
- fixture hash: `sha256-0ad9df6b8ae5c607f68351ee7486b6c576a76ab786cd98ba2afb2b775407f3c6`
- score hash: `sha256-ba6a8213193cd6857038fb20fb80c6fa35077bd5606de0013da20cf310641b73`
- bundle hash: `sha256-63bc56f55d19a4a2e14e1b84ba8970e33d766d86e109a885f4a321c4a30b4a34`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-94660750e35c615fd2752347aa8a9c858360545c7c42d4d1838c4d6867570823 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fa7a73a101ed6b014b9eb0368b996ff7f2f7ee187ca41f2b02a97a9dab4d12a1 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a3fcf8cde277624da86e231a80b88a3194cf2dddd1090f1b3882d053107c5d64 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-07148c42464cdce78e9bf3819ae54549a3c48466d30a34b08618c5dc00c28545 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-efc16e1b | sha256-9af108520335b77f42898e3739182695f0fdeb93f9a5b471323b03e0cce8a4fb |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-efc16e1b | sha256-18753abb9c02778b3f2f70dbcc3f0460c8bfc671d56a4176501794806c9e17a3 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-45e65668 | sha256-4362cc2462c1faaa0be88e9ec375e728943e954edc8a9690d99f74b4b7e60bdb |
