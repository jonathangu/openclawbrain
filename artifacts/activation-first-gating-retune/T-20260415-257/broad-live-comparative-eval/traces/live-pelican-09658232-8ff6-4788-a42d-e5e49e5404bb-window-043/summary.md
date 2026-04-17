# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-043`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5dd6be200875e55287b6b027aec6adb1211d2b987b83c6a37b985516f7118529`
- fixture hash: `sha256-eec602e66445ff4dd47c7240e799fd3d8564ee87f3fa97f5e6b5673abf356c14`
- score hash: `sha256-eceebc35546c0d9efb1972ffa9754de46bb1c1b1da44d2e187e2feabadf5ea88`
- bundle hash: `sha256-c3ff4c0c35bcef5c648eebfc19d5be4fccd5a8f8c0805acd7972f4d2d9ed937d`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d99ddcb432bf41d8fa10f8ab6904c40f835adbab6565ab293b9f4c7f5ab02130 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d96620753410722741f8ae88aac1f79dbe3cd15881cd722b26d1f5f73301c2cd |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-08c3c71f0eefeadc8b002c40f1e31ead31d2aee52ca2a10e8e22d5e1f8af6d37 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-3d0fbd6264de76ba29dd4372fc226d5876d43cbd390431c233d7c8861af63766 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-24fa0522 | sha256-5162e1745e502c9abe7ecf61f93e538039defcc813ae4bee47bab3deca64a692 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-24fa0522 | sha256-361e38318518a7b36849ce30e67fb84b4adb3fed26488b59d83d9ea72fa0b768 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-2a207643 | sha256-b765faa01845d359b9f9f1cef9efc2ed07eed91df3ac9fd80d6c3c9fb73dad5a |
