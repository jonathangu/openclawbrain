# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-15894236758cffd6885df088771bc9158a039d8e6dca7ba37e0c0ae93f2bb22c`
- fixture hash: `sha256-897b7fdc496e16305fc54601a8aba44f23b5322a6b7036c26e9f447dc3d9e950`
- score hash: `sha256-ce6233901a3a0e28c7e7a5e6b6f2397acc5badd965f141c2dab7ef8645c80d01`
- bundle hash: `sha256-7e4c67640810a70eb005ef1bd64dee6a66b18467cfa105e09d3db998b829c4db`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-05b9f1a0d0ad4a80c5a15a8f7ef9c5d2527f8753fe005026d39ad6af8199556b |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-309d2eec8071858ee490881d807a1afd37d7c912a9aa94b88092e8faaa9bb7fa |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-98033d02064b32c3fbf727eca2267acda28869ec4b7e3e7e2c97bc667d926f97 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-e79a38228fbb4b41ac5d84fc0e41aa33204b2fe15fb565bdda0c9b023ce2ff44 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f71ade68 | sha256-dfbcbfa2c32f442ffec8b6e2bfe505e9d13e1a95adca5743e620918fceac03da |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f71ade68 | sha256-4e88a263cc31ab52d12f07d1496ac66b487348ac951fa158fa2e3034a074724a |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-bb9b68fd | sha256-6868afbd4759bd76402c0778790e032725534d80fffbc3e91fbf22596108b0ef |
