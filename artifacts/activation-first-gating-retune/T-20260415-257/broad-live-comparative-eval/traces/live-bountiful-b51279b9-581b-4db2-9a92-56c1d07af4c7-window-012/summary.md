# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5f94013a0a094ba3fb5146fbbc0c01b19cd73626385a314188a4301a7c00be04`
- fixture hash: `sha256-113ed14948559f61f0991314db5ab7b153e15f743e52adfd432a03d575e47935`
- score hash: `sha256-00a235e8a58f1422c1e97226e2c0390d92e56095bb329493de78fdb025c27a76`
- bundle hash: `sha256-67652a86f020e79acc468c9afeb52708734577927b057a9a2d73e64c9895803b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b0dc5c5b8bcb0e2bae32d52cdbc81c4d2b373af818d11c1ab51e1555491f474b |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-62a8084081389f8c5d4b55c5a74f3f9d518ca1f04fdd7aed50af3a0199da8019 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ee8c173d83ae08f3680b55f27e152c22977845d458cafbab9a0f06a51e350b19 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-212503c2d2b95969216f4a502d3598af7c62cd6b51061fbf3a7e003b75f03ed9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-cae6235b | sha256-c64094c09dd66a10361c433012181878903226033d22b841303d7e5be70c8d41 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-cae6235b | sha256-7fed248fdb69c9afb9b92623141d33137097baef62adfc13c6f8c69acbebdfaa |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-984b8f38 | sha256-fc86e20a7c12aa99453687bd3ab47c677a8dbabfda6eea608716cfd83be6f3ed |
