# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-08b73891909d4362ec26f6fa9db500532bf1bc8c805c846177530e06134e3890`
- fixture hash: `sha256-8b9b0ca98fc7faf65751139ae1faf124a5228fc02a0f5bb6427265ff145c7a87`
- score hash: `sha256-a3cbb206f86bf8cd81efb3b1e0df22bf21f0a6b9f6adf005c8d95ca242b344c7`
- bundle hash: `sha256-1f6dafcba837ecebc24e9d8923822fe7f3562b25b4cefe09bf6fc2bf8d8155e0`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1f871e8b1f54b24b9e075d5f4db6f8b41f6cb53e929f6d747d42ccbb2426d8d7 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-280ecc321791690b2ec62bd214c83abddf82e0ed6d2b75d12d5a38beb0816dea |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-17396375439deae5ac3c57de17cf81aec2a23b30fc62d5b50b6d060458b128b5 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-c0dba1c1667055c61cf6a8044f993436a4100bd778a8d994aef8122d9325e4b9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d547a38a | sha256-0e338ed802f6629f3c49f7ce1a82dbadaea83d389467ad8557ad9d2ac0b4de54 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d547a38a | sha256-c3edec17aae01ffd37c278f7291b3192110c43f0014ac50a652ccf39ef04efbc |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-534ab089 | sha256-1d66a25fc3c32502ce50a5ae4d254f506a02e28541c5923f74010b649cf3c706 |
