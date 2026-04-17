# Recorded Session Replay Proof Bundle

- trace id: `live-main-7498149c-ca61-4cda-b16f-880f2c1cf323-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e725d27535b9af2607edfc55a55297b3d0ba2750ec3f143ed7ff11bf77f51432`
- fixture hash: `sha256-ad7016b7c1b3014d2314a0a7c149b979d45fc36075f6fdac2c19edc868ece1f9`
- score hash: `sha256-a8de3e1e3dc7456a265fc0791f6e929e4e7294b82d7c1f19d7ee5fc668de2480`
- bundle hash: `sha256-1cb848e7274755fc7770bc5286113c8625e8d90e9eafb89b0341397c92b58b5c`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-876d8e0d55a1dbf8e84f9354102b290654bd4fd9a97c3f26b37887de51274c7c |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5c4373f8699259d61fed6d524145f5734236b091df14d002224a0c178d97af95 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-819fcff1c2477d520d807edae765d86c07e7030bf1ef4672bbfb97c96d16e7ef |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-9cb1f52ee3fb044e7cdb07704c6c1d36da9a9024e61f4acbc43d42f5d05f92b6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-54671da2 | sha256-8a9239aa82f488e17e0f4062b4e0ae33d402fa8aad5fb95670d5530b20fd0b05 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-54671da2 | sha256-50e6656aa6fc3ad541eda4ae0d6136377d0a0c6cef6bf44a52dd9266c867b116 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-c000f177 | sha256-2a845c2e3640e98a7ff47c4e1a908cd57d225df526069f679cafb6dcc5f7d850 |
