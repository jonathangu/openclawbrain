# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b6443732e15180ade7a5127a9fabf70925f6863db8ba4387cc6be009639e6099`
- fixture hash: `sha256-0ad9df6b8ae5c607f68351ee7486b6c576a76ab786cd98ba2afb2b775407f3c6`
- score hash: `sha256-dd4dd8a90cda19ae87503fce20ba0a64943e7b0b17cb14d02f65efe450c873a8`
- bundle hash: `sha256-6873378caaa03ad3b1fb590a7df3935c6f730c9bae9588c475687b04cf587cb9`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-94660750e35c615fd2752347aa8a9c858360545c7c42d4d1838c4d6867570823 |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-f14de095631c4c38de0a095260e32fe62fb610e6a31eb5605c08fdef844fcaca |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-8aaaba84948acad7b4d3da36da1741246ea925065bc4ac5eddad910d80c8c446 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-3c3ab6bd840af73539ac534d3dc63d5db783fca05b0ecfd6299e1a17b6241794 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-45e65668 | sha256-5fc5fae39ac22600f125dad137fc823785884e7d2dba465fde8533ac6aa846d5 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-45e65668 | sha256-22223e32697fdd800adc68cf7dfc3fb5c5c8f5874e9959880b1a017d6b4296e9 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-45e65668 | sha256-5fc5fae39ac22600f125dad137fc823785884e7d2dba465fde8533ac6aa846d5 |
