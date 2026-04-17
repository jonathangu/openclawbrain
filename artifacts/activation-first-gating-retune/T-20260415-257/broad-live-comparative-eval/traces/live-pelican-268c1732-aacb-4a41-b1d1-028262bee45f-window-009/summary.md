# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0c4b362666455e64ccec1b7026696e6d7ee86b07af9d91203554a5f880643a7f`
- fixture hash: `sha256-ed322207ac696cb8afb94d5f75c53ba4423b96ed55d3a35abcef96ba37d6147e`
- score hash: `sha256-32c87e5eda58c8bd5babde3650f2d7b38347af39c1e1e248dab3f71a4209c1e7`
- bundle hash: `sha256-bf533c6c0b2455731fcd10352cf6728751cb4ed884ce431612ff99c357f12ee4`

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
- phrase hits: 0/4
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-659f0a773cf1e71603ffb20a5a28aebea6d5db6139dfdf0255a605e7868cd22c |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1124744332356816f0237d2f90468ab94365f3033f1f2886da9dee1b82a19780 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5f1169eb29a647a1f398a2ffa0bf142266d8c8e061acc3f82d0af219f4155e42 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-ed2f20354058bd88388caa84cdeb83b6dfd0b3c5f3d9f7cd824a3123238148ce |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-586c42f4 | sha256-116d7eb78bf712c967e34ab9a2880bcd5738414400d98c221a16ff0793002ab2 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-586c42f4 | sha256-37af6286d569e0090efb78195ded76900d4b362f581c2fec49e78436486de92f |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-26ba0bab | sha256-67de63707cc49bac44f5d09698d9197351b133702741a5dcd94790c799b7ff17 |
