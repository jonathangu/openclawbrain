# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-08b73891909d4362ec26f6fa9db500532bf1bc8c805c846177530e06134e3890`
- fixture hash: `sha256-8b9b0ca98fc7faf65751139ae1faf124a5228fc02a0f5bb6427265ff145c7a87`
- score hash: `sha256-82edb836de3d76d6aaa21c6a75c097a12a92864538d588ca4125fb784f8fa75b`
- bundle hash: `sha256-7c119842c47ba1925e97bdd0047fc31156763ca7288763c243d06bc428c84dcc`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1f871e8b1f54b24b9e075d5f4db6f8b41f6cb53e929f6d747d42ccbb2426d8d7 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-18a93b60f77306e05e0477208c37785c2f709828fb240b7bc6e5b349ab582775 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-458b08e84e685cb8a1ae546b38ca763a06f31e9e63e5d21ccc9524145b21872d |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-fe073f95173d9e6fc6450d932a5f60dd70e5d9c2164e3a0493348505432ee95c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-075c5e35 | sha256-9dad517683be060aba56143aac2355005a571745f46e0608a1f7ae227b7b666d |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-075c5e35 | sha256-50d9858d88ef2328b80e3149527dbd3bea88997676178528a2566b62771de983 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-855f6b34 | sha256-2d0cea63f598feed6987a65cb1ff6da8c6af1be6a0b6323524b8dcb3dd8a6ba3 |
