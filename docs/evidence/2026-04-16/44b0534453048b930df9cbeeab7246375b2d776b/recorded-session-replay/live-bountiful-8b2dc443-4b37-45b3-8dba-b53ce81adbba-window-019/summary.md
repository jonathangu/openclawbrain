# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-019`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b26af2b1e36bf39a5b818412cda88ed6aba582667f9a54ce799e21e291662727`
- fixture hash: `sha256-dc5c60cd5ff0fd0eb8ea43eb629625260e32638ca4678441b2528e3ed52617bf`
- score hash: `sha256-50acb5a356898f8d20d302a0d85c0b5e0c2359991d276e6a482f9bb6b963c9f9`
- bundle hash: `sha256-be78636cc92ee35ead2688dec6c2183cc23ebae727ca3d52888f3adf4de77e30`

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
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0230f532bcd3c2592f949681c166dcb99537a5fc4f8a4279b73154c1cef6a61c |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d6d6c805fe6c92cada6263c5837365faf37ba6fcb7cfcc120ae818b9b38d1a3f |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-d1121e4d98a086aae952567a690f45ee4f9e7becfbcc5246e33d6cae904488b2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-273b80ae | sha256-8eefba68c0b723bf073f0cf57d8d27b978152584e857dd07aa27c21501ff26e3 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-273b80ae | sha256-c3f9b36db28a5278870267aaec6b213e50c1a8be16e90a56135138f2f3721465 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-04b4ef5d | sha256-cf4fd7314c7d06475e64487cbf0004070ca55ec15fb17017aa98d30a7acf1183 |
