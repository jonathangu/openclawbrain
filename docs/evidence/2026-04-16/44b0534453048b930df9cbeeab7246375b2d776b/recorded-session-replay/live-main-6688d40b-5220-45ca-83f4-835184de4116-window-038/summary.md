# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-038`
- winner mode: `graph_prior_only`
- trace hash: `sha256-97f203f8cf6e54b5353937fa7c2a9de19dc80e0c9cd1f7a7efe923af5d952db1`
- fixture hash: `sha256-125dd958dc76d20d11fb8d8f175ae0fed91a68b28d91d5171d2943217403837a`
- score hash: `sha256-4aea0f9ec440cc2e974ade9a6510b83edc76c0ce8fab8b29f2443f79bf8c0b66`
- bundle hash: `sha256-fbfa962a6630d57eba3c31ea3eb5bed370e04e60e54f11c3f5b427f8fe689a93`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-511d0fcc64563746c6c18b192b94f28492b7e276c306dddc6df1e62d381fef89 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-17e76c359f0cd811737121fa8d075fe7b1087580a209d3a6f34b086e68c039c2 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7f8f6a72dac49b7bd457a0b0886957df7058df43ccfa3503817402899cd3fded |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-e6ed1bfe72f1e24c79d878c1b7ec045c3efaf2e2debe65535fb23a5d65fb97d0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-afc66e73 | sha256-10160c4df359e8f29422f875af0d277d5cbc7c4fd501ba4e038a0172a98d3f46 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-afc66e73 | sha256-751109b14025a6f75a8676c4d0397b4fbe33ecc9bbdeded01e1e0143d9f47cfb |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-be000d54 | sha256-c44882d7fa591ffdf4e7dd9cd16e8f5a74d11241a8471595639671101cd3040a |
