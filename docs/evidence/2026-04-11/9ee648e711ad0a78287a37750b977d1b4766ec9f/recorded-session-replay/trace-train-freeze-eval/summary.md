# Recorded Session Replay Proof Bundle

- trace id: `trace-train-freeze-eval`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4f728c53041fe80cb5bd8e01968dcf8bb4012e0baeaebb1be20dc5d213fcde73`
- fixture hash: `sha256-222f49747a6fc48b5e1b2d503822eee5bc04db6bf6b0ff996b0290937143bc04`
- score hash: `sha256-327801c8d40cf248a5253c64c8a99d6c24d3b2f56b44bac64c95cf52d40854c8`
- bundle hash: `sha256-58ba7b8463aee580543d2261b70036a9c18c766b22758afe5f4ff3bb8271e248`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 9/12
- compile ok rate: 0.75
- phrase hits: 9/12
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 3 | 0 | 0 | 0 | 1 |
| vector_only | 3 | 1 | 1 | 0 | 1 |
| graph_prior_only | 3 | 1 | 1 | 0 | 1 |
| learned_route | 3 | 1 | 1 | 0.666667 | 1 |

## Hardening Snapshot
- compile failures: 3/12
- compile failure rate: 0.25
- warnings: 0
- promotions: 1

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 3 | 0 | 3 | 3 |
| vector_only | 0 | 0 | 0 | 3 | 3 |
| graph_prior_only | 0 | 0 | 0 | 3 | 3 |
| learned_route | 0 | 0 | 1 | 3 | 3 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 3 | 0 | 0/3 | 0 | 0 | 3 | 1 | 0 | sha256-d0d7ec5bc630c9366bf082fe9e3df18c7cc3b50bc509aba5c3b112c7f54804ea |
| vector_only | 3 | 3 | 3/3 | 0 | 0 | 3 | 1 | 0 | sha256-6a4077639a6cb6cdef75bf75b2e234c3b4a024cfd62f43545b491cf20604ebc2 |
| graph_prior_only | 3 | 3 | 3/3 | 0 | 0 | 3 | 1 | 0 | sha256-062e052921ebaeb27a851204c2b07187db46e8bf1d5af36ca4539b74d48abfad |
| learned_route | 3 | 3 | 3/3 | 2 | 1 | 3 | 1 | 0 | sha256-42fef7a338af042793f1651750bc763c1e9547e1e6e80cdbd1e7a7e616edd9ee |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| no_brain | turn-2 | 0 | no | 0/1 | no | no | none | none |
| no_brain | turn-3 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-e519d7dc | sha256-80c4c5d184dd90c5eefde8626b0dd3009e20e2e04f71bad8d370264944ead3d1 |
| vector_only | turn-2 | 100 | yes | 1/1 | no | no | pack-e519d7dc | sha256-89bb75bfff0ff538dc46470e690b5370cc7d6a56726acafd9ff47636ade3456a |
| vector_only | turn-3 | 100 | yes | 1/1 | no | no | pack-e519d7dc | sha256-80c4c5d184dd90c5eefde8626b0dd3009e20e2e04f71bad8d370264944ead3d1 |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | no | no | pack-e519d7dc | sha256-80c4c5d184dd90c5eefde8626b0dd3009e20e2e04f71bad8d370264944ead3d1 |
| graph_prior_only | turn-2 | 100 | yes | 1/1 | no | no | pack-e519d7dc | sha256-89bb75bfff0ff538dc46470e690b5370cc7d6a56726acafd9ff47636ade3456a |
| graph_prior_only | turn-3 | 100 | yes | 1/1 | no | no | pack-e519d7dc | sha256-80c4c5d184dd90c5eefde8626b0dd3009e20e2e04f71bad8d370264944ead3d1 |
| learned_route | turn-1 | 100 | yes | 1/1 | no | yes | pack-e519d7dc | sha256-80c4c5d184dd90c5eefde8626b0dd3009e20e2e04f71bad8d370264944ead3d1 |
| learned_route | turn-2 | 100 | yes | 1/1 | yes | no | pack-8a7e18b0 | sha256-b0837c2b50e3854aa0747d05e61325e95e1aae376a3916c6f8284edf821e52be |
| learned_route | turn-3 | 100 | yes | 1/1 | yes | no | pack-8a7e18b0 | sha256-fd54b93d10bd329f8dd29e26c5b26cc30c19c77013df7ba6dc43bedf22d28055 |
