# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-010`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ad0e2983c4f13057addac140e7ec01136b02517807f2983a2a8218c39f77ac60`
- fixture hash: `sha256-5465f326b57b932fce5d721740c2e94691b72cdbb86ecc6d3f5feebd376f974d`
- score hash: `sha256-cbc1f7939e59328e6a6c0b3d2bf5447e10cb1bc46b6a85829499dfecb4b1b58c`
- bundle hash: `sha256-0daeee616544794a5c76dd2afff443429394d5500a6add5e90816db0cdda3448`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-11232886674dc9e702bf2807ba0bdbf15e55d8f77564bf0f29c440dc177c94e5 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-36be6dec52a5748743e6b46c8f572bc00427019db902e30e53386dba80578586 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-15d4f3ea2e2accc37c6a62f976c4a6e6abf8dacc33238d6abd87d5430c93ae8b |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-4ce1887eb00a9424fb91b212fc3b479667f3f21998827746a7fa64fe723d5b2e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-8d6fb071 | sha256-fad592173ae4d96691f22672bbe10072b9e2d37288ccc2b672a939ea1b982d45 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-8d6fb071 | sha256-c6c129d75003ecb5362a4b39558b32b6b71d0b7ca6d37a3538d4803cbb9b103c |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-509750c0 | sha256-0ef7544ebaf1bd4d2e1c06327df13f2302926d4236769b081b23c6ecc62ad905 |
