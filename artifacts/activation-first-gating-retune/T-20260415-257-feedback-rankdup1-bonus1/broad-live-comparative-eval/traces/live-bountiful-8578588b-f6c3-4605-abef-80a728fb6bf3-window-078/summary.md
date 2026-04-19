# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-078`
- winner mode: `graph_prior_only`
- trace hash: `sha256-09e16d33e75c4cf1bb693ae2b746367b7c597d6c1f3807bc023d0caaa70d4704`
- fixture hash: `sha256-5891ea95069e0741e64b63dbc158c08dbbd916c4d462cb19e66a9822069e3b77`
- score hash: `sha256-30671e2c9d9ce42ac9671df5c9297be45f948804361ee7f01523567bf2416bef`
- bundle hash: `sha256-354bb70cc175f58a31055edf61fb75663a01f11ab9ce6d4135b55faafa0cfbbd`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c24d7a3f72e63cefcedc0db743257682eea37c873db139e3264a8ee79b5194f0 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-5fd2345c675ab62f35ff399331205b3c34bd5be5003c008098da2df4e7a63664 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-8d83daaa538e7fb900cb8dcbe8868cfc8ecc33dd689d962f42ca5a10ebb8f41e |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-9201bc24ecfae5f5f5cc33474d6e95fcda8b547e30278d73cf257c75a4ef5f8c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-5531588c | sha256-22117ce8f6108da9482b93972db2aa5c181135fdaa53f57ad26dce20654c659c |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-5531588c | sha256-259b63954b9767f4e98865e39f6dd233299f28fb0f4813524e99e00ad18b9811 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-5531588c | sha256-22117ce8f6108da9482b93972db2aa5c181135fdaa53f57ad26dce20654c659c |
