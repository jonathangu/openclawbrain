# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-032`
- winner mode: `graph_prior_only`
- trace hash: `sha256-cb3d634932f1bce6d4693c067f779badfb747407b8de4f4dc108015f5fd2e78b`
- fixture hash: `sha256-178c9c975a3f9bee04b778ee3424e4eb908e1106cd7f867502edc61b1de425cc`
- score hash: `sha256-cedd7d9a089363933a40c060ccf3853d8397a1ae1cfc5799aac73d34ddd2f997`
- bundle hash: `sha256-7ea5b0fe5898d25f2d58e6327f4804af0eec488219785f41a068237a2aaad2d9`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-252786d108eae05056482d31aaad41cb1fd7abe9a8bca72a4a7a00c78ba84b59 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3e63e465b9475d3c7ac5367d0553aadcd51f1733017abdb999ae49edc681c31b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0256798e651bd7ef4b5e39cdf210952a053f66ca560bfcaf0ebb4f90ebdfd98a |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-c8444be32040b0d7266a00ebbf65fa2ed4c61ae4c9c365c207773abd7318f654 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2706bf25 | sha256-726a0d41b48ea570b538e7fefd6044ee2e82064a86d1670ec0ffa6a14ee76cd8 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2706bf25 | sha256-ddd9817e1d8dd180c0918e3792783ef12ad2b11de8d230749f7120081bfc3b32 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-2706bf25 | sha256-726a0d41b48ea570b538e7fefd6044ee2e82064a86d1670ec0ffa6a14ee76cd8 |
