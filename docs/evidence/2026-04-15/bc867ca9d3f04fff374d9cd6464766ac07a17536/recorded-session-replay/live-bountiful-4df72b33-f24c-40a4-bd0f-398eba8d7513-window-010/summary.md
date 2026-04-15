# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-010`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f896b3d7889710642e066f81d9ef38f09a0375e7c4550a3de44bd42b8be0c728`
- fixture hash: `sha256-02b10f8d55f27089a7a2cdde95f78ea9472dddb1b95943a1431bda089a73cd5e`
- score hash: `sha256-fd0d328da795572f39b919b335567af8dd230b2598fdd9401fa7a5cd0a7a0c4c`
- bundle hash: `sha256-acc62b67c27fe553bcec364213cd522c801152279b3f69ab6830b1b6699ddc38`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e6e466bcaf85a4528dcaf1f22f57a3cde69a22135dbdc628862617cea9e4f77f |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-922a2bc1993d9039c89cef24eff9fb5185ca5ea864e91cd248edf9fa239dd232 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f538d6914778035864e79eec2c8ccaf1a6c2a03649ef9ae638386a620ee1daf9 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-c381b6681dd92857f39ec36832669a59f8daa3313409c8e49fae979adb8f11d9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-3062500a | sha256-74cd913ea86ceb435081eaa6b7348dae094a07d8675db5436854969f9dd08437 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-3062500a | sha256-2ed6847c86784d652a8f5ebe3d4b49eb0a13d1ba1a7055a77fdeedfb3969d60e |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-3062500a | sha256-74cd913ea86ceb435081eaa6b7348dae094a07d8675db5436854969f9dd08437 |
