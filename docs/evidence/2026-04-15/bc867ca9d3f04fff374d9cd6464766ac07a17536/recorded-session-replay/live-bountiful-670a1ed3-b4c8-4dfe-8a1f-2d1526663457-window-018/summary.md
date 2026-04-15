# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-018`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8731aca670fb1adc2a11de661b208e90de02229e43a59b819be0c26634995543`
- fixture hash: `sha256-b091c6d75f126cd4fa41e0e62e2c1bde2a5cadf897b977dd808714e16a9eb7f9`
- score hash: `sha256-7b2bd1ff22e4525d3233e4a5022cc70cce7971ce24f7f91a3e2e59fcc7fdb9b0`
- bundle hash: `sha256-9e674fd6af584b7b8f495b2863c875dcd5346e20231813fd948f8cc90e109057`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4dd25e120884595a4500dd8027a1e5e49f93c256e2e2739aa127521c9309576c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-dc87a7eae71b7434b7927f8d15279a2769a700580eddd3b5fef3a5446ffaeab0 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0f8130b51206a14581d1d2ce78f6f71abdebe7ded6f37ecde2c8c53106602f34 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-16874191d7da227feaa8204cfe8e7b8e0f9b93554638fa9146087b38e3f2c6bc |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4b557c63 | sha256-755f2486f77fcb5003453f413694550c6506f7c6e93e1c41cbe280c869652533 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4b557c63 | sha256-4adee1505dc8d89b9650c1ff57f35053beb8793c5d1f0e60f22737e88d6a833b |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-4b557c63 | sha256-755f2486f77fcb5003453f413694550c6506f7c6e93e1c41cbe280c869652533 |
