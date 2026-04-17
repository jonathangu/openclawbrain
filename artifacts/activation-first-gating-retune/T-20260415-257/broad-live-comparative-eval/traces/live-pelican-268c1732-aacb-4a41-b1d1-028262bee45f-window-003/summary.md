# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-01a3188870712e041c94dac038c1913b6a8275f11c9b961a30d44d4a9193a2ad`
- fixture hash: `sha256-61072aa6754828e3628c89803b9d747baa926b5fb67bd06b8dcc6e5a7d888974`
- score hash: `sha256-7309b9e0501699d546eef899b0938ad46d910adbba99602e60f9c40f42815b5f`
- bundle hash: `sha256-54a95d52a81a8a8388daa89ba85300e388a3b5f31d53ccad43570b3c249289de`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2838eac92b5037b9e7a88f6a187516f6cafc0d1eb9fd70438eb0e0126665d9b4 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f13a556341eaedb9e58ac5c52f897913ff6977cf3f5a79079bae87d86c0d77bf |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b405820e864df364fd0c60e747f6e89ff9477c0aaf6d86a646fe23eace33436b |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-e04f82f690c06d51174b8aa47bbe5989a58a1ff81c39f3e88c91ea46455f16c9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f3ab826f | sha256-6277e0af355adfcaf35b1a4aad0c4b7f76182317cde3d726ba68c44b5d9ac13c |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f3ab826f | sha256-97ee2376d984580ae5f9d7f5178302b0a467c0161abd49d4b26a1315f4e290e4 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-f18f8268 | sha256-0792ef46a67e92a913da0cfeefeba3b0ec8768754e4480ed109a68076f7a6092 |
