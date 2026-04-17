# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4b9d58238866be4c30cb67001ed41c476bb074abc457ce91f27bdf2a95087dda`
- fixture hash: `sha256-93a191f41c9134f7fb1b39f4120c598d79722f0fdf720a1c60726eeea45f85a7`
- score hash: `sha256-bc741dad2f13c967bf16c65a906e7628d410796ff701b6aa4f5b763870ebc55b`
- bundle hash: `sha256-e3943b78adcc2ec5b67de313b880a430b702ee339d752632a492661536eb6d0c`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-53ac170bbdfe31610a82a7fea6a20f739ad327e9856e23aa713b46f86601ea52 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-1daf2335803a7b09102aa699a9515e688f86869906b7981b00f37f686cae99a3 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-818f0d6b4f4b82bea18d67f8eff9db2d446084ef2ada4d0d98ca98a8d54688d5 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-b14ee74e8ffeb353b27d635b713fb0b40c79d4efb7c0042da57adec339e59261 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-9caf8fdb | sha256-2599f735267cd2b1f2607640228c58597b791f08891fb3f14e55cde55c299ed5 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-9caf8fdb | sha256-dd8dfa6d84f27c735dea3f76546c762f826c932a0a4588dcda6c464521063501 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-7b275eb8 | sha256-b68bbedb53d63bbf1aac3c874c03733b0385c64033b6d8056ef22d5a19419973 |
