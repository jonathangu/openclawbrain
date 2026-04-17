# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-019`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b26af2b1e36bf39a5b818412cda88ed6aba582667f9a54ce799e21e291662727`
- fixture hash: `sha256-dc5c60cd5ff0fd0eb8ea43eb629625260e32638ca4678441b2528e3ed52617bf`
- score hash: `sha256-d832c60235519ca7c6058dc8a5fbcc06629f751e36a94d36bd6c202befb35a5c`
- bundle hash: `sha256-a663f0f893bf676f1fbd2832a4cf9891cacbda76ac70d76eec5f62f408a538e8`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8233e1dc85a16271682dd831a32fd53162f821cae19b4a63ef88dbd637e3c9f6 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-588248bcee46e992ecf9536f2c9ad6ef61cf4fcc442198658f6641446ea33389 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-51a3b32450043f8e838e274aef28618151ca8932db1afca6727765f914979fd8 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-09ff3bb4f85a37471456b61f2f12f484d1e2806e2ae42cdd0f9da08e2609a940 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-dc6d24ce | sha256-5de7381ad010ddb785aebac0f556365deb133fd4e369fd6ba9bbd8a40a0cd073 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-dc6d24ce | sha256-434eb99f56cff64e3ab9dba4971a6c10a0927f7d2fbe31fff32feafc5c34fa0d |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-b9e6937d | sha256-9821e1ab7c3ed2a58ac18dcd1a5d480fe80bb4cff872adbb4f15c176f207a425 |
