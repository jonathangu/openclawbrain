# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-068`
- winner mode: `graph_prior_only`
- trace hash: `sha256-96a8ce03d5ccd3eb8ac2b891590e50e52a93852936116375e423e4aa54e6c87b`
- fixture hash: `sha256-5bbc260fcf82f9c3549879279567f4904231f7dfa6b4db116db4ff63f77dfa74`
- score hash: `sha256-51b14c7df098efa1c7f230a18bbe1781f8f125a6cfd51d37fb26df23810a9a3a`
- bundle hash: `sha256-b3b20c11bb457e1b4b87a18cd068108034939074c82399cba1f0242a2c79d754`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-65d66b9ffd94fa10d6d8c747df6232ecc791c54f9b91e74c97206816ca5781ad |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-bce5fd0638434479c9e5f3be95ba45243569babfeb2996999ab1fe666ebafebb |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-3b3dd42bd0376a7e2d7161581d0264f2140bdf9759a6fac760ba3b9a4a59096d |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-59a23eb42d514892ec15633f25efdc70b08494f3c200948b58cf3c52b1444e4c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-3aad19b4 | sha256-0429b9aaeac97c9e601c765b6f3964521e990a40f90f769589c67ebcb22db6e5 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-3aad19b4 | sha256-52f03aa67a1fddb0e31ef385d772fa77b76075d9036b4098d9de25f828c02184 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-3aad19b4 | sha256-0429b9aaeac97c9e601c765b6f3964521e990a40f90f769589c67ebcb22db6e5 |
