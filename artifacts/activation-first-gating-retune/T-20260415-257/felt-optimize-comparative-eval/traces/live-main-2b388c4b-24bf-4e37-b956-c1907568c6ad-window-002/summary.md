# Recorded Session Replay Proof Bundle

- trace id: `live-main-2b388c4b-24bf-4e37-b956-c1907568c6ad-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-a7dfee8569cbeebae47062b33f26bd559b44ab7e32ac7a65f3d53fcd4f9d6446`
- fixture hash: `sha256-bf2c49e43d0148934d94e443780f19f84be1befb9f46554500ee32090d69fd0f`
- score hash: `sha256-9d889d7f8d6005886f00f448d764b5fbc2ad65604cd2ebd260506ba8d87a0e70`
- bundle hash: `sha256-bff36b56aded34f1316330a346ab2e78a11db9b9bd70c9d7deb78a4d345149c1`

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
- phrase hits: 0/8
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-aefc644f475f6e64faeecc10e1bad33424cc557b74533b3b9b16e76adc362925 |
| vector_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-1b4f1daec4c05ebdb585e87f863056f57b3771c3953d4dbefe082be3e75d5abd |
| graph_prior_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-326f79f0e07e8cad7bbcdde8714bb5b27f9e56f5c02e043aee3cdd781dbd910b |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-d9fb23bc6b111fa9ac9ee004296001a2b0f8bd9cfd1ae90aac2716f284945f51 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-188b46cd | sha256-d74f866f41621a69b41edf90e14c74a39c70d28bd8fca729faef4e4fd3a5cd62 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-188b46cd | sha256-36b6dff4d11635f6475f9e8c0aae973457ce901fd86c6dd180f071b2233a996b |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-188b46cd | sha256-973722839a130d9505b7a8b1603472ba5aebb43cf87c23473b56e6a2f4e3eed9 |
