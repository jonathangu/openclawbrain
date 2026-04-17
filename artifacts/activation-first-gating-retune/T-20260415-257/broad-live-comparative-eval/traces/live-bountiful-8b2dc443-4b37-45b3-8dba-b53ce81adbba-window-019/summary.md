# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-019`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b26af2b1e36bf39a5b818412cda88ed6aba582667f9a54ce799e21e291662727`
- fixture hash: `sha256-dc5c60cd5ff0fd0eb8ea43eb629625260e32638ca4678441b2528e3ed52617bf`
- score hash: `sha256-9aef30dae145ed34a459dc3184920ccc5ff6f00e712069afb628af3db5d9519b`
- bundle hash: `sha256-2b2ac9c7c895e11c2da02e71f155fc32faa553fa15d7b91a4223a4af50713d55`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8233e1dc85a16271682dd831a32fd53162f821cae19b4a63ef88dbd637e3c9f6 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6e8f5a2324445d920ac3696f2ab969d5fc9a383a33d58f729718c2dd27e3df83 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8494c435591284b8155b491fc1818ca912d7cf6c99c0c371263d7579762883d6 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-42979bce110c063dadb5c4c6df81c4284d5bdc9a426853c53ff2a704c3b09eae |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f16dbe45 | sha256-623bb484a37b253a6e25eb934dca50df7b3b6629622fe56e4562badf08cd96d8 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f16dbe45 | sha256-07a320c6a927d04d0c861dbd0b3b350206fe7d525d07a8e8c34dbccb5ad2db65 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-cee72cf4 | sha256-56cf01cd90fb5b507761da2cb799ea19f441f71001764eead76af77dcaaca64f |
