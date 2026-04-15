# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-185`
- winner mode: `graph_prior_only`
- trace hash: `sha256-24e1e9ea471d19d207e35c598683b69d84119849186e1c11e6ddd97932c4aba2`
- fixture hash: `sha256-11642bd40eb6fe8c9d53921bdb1bbcbbdf6e5f35f00a6469f30893bcfb466a96`
- score hash: `sha256-55cda7ba26f4671a54af00cf52896ee9b778f4176a956dd12a960e2025abfcaf`
- bundle hash: `sha256-fd8207d978a2f671ccb07710de798af413cae32cb37062bc3eb60f38eef68582`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fe10a677dfb68dc498fb14e838ed3e08e036ad9f9df81513ada323fcaad39838 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2affaea163b296218bd05f356bcf6beb634e21143da21e92330595a044d13cd5 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-bbdf4974824fede4e74f3235a5a1763feae12e84b1478af1c4e085590175b8ab |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-959a4eb02caf44f838aac724c307dc86101a0710405ec09ad70d4aea793fd88c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-90f79d76 | sha256-2333f8acbf4bd4dccff6394022c56b71eabb09f720dc19199e63161cf7a47bcf |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-90f79d76 | sha256-b473c6c76ea0c901ae3e9b0f888c6b3fc1bd51cf170125c53d02df1e85ac7e13 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-90f79d76 | sha256-2333f8acbf4bd4dccff6394022c56b71eabb09f720dc19199e63161cf7a47bcf |
