# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-027`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3f260af2c7b68b1309e9a87df75f2e99f6d28d47bb3f82fdbd20cd787e51e3c0`
- fixture hash: `sha256-4a50ee1d4a23bf54584481d6c799516fa1f1a51aa4c19299da0f6a6b73848dff`
- score hash: `sha256-40b2f6c66e0c5b3ac4193e848a83941b0c9efa7d0b8300cf642c5369e7b98cb0`
- bundle hash: `sha256-f51955affffdb1cf98e73a003987bc65bae01a1b6e8e0816e7bbad22c84d7b11`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9b2a597464226db9617a3470772ef24fd543ab0477b7bbc0a0ad5adf41bc0dc2 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-219dd304bd3ec7599d083b1b376f88116794e32fcd9f52243d14f330f77ce637 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2773cf9296c7feecf4900a5d2c4de23955c406b4bb48d0f4352399e58cc91d77 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-557dba5a924dd055a879f038bda176f95a007a0b18e10e205d6935777d48d73a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0f380720 | sha256-abbccd376d03e6a0fb5c62d15394edce453c61955e328a3def58d1f98681e52f |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0f380720 | sha256-ac00c5e3c3af546fea2f476aa288eb1a70cf1c385d48511f9e8818f7927aabd9 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-f7151dd7 | sha256-70ed1244e82c564d91185aa15680d5904e4db8487c0954a1639de107c7711ea2 |
