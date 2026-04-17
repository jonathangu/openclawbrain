# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c178ada-4f98-44da-9ab2-6ca13f2e2441-window-001`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5c1146574706ec395e6f5011fe3bdf3e510b31ef69670a55eff27bc156061d1f`
- fixture hash: `sha256-b0b87869202da9099b109d7a7b86f16484e8b3960b663b22dcb9b0c0fd925784`
- score hash: `sha256-3c4b690e3394b05f3efff4c3badb7d44dbf497c2850f60ccbb4760d6ffa61235`
- bundle hash: `sha256-88f51de457434046b4e541c3e4b9d82b3c61773a8dadfea8460cb4fe1677c604`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4ee8a7a4748f6af35d73940b990960d0c8506d722d1756ec1464f9fd52079a6e |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4b6cb5204d8a65cdbf57d95fc65656a71fa53dead0883a73ae229e4f472abf01 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e63cf8268cb95d09244efde49a11c9fdaecd0218c794932ce9395e2197a51dc9 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-c6c1f331b3a5c33e0d4b195eddf88f9ffc61740979d5dd62d98a9d80635ab755 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c635adc8 | sha256-fcdff6a9b73a3503e662870b892d1597818855c5c8afa0677097e20906e8284b |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c635adc8 | sha256-61c2953d964c968df549a32fae5510fe19b157776580e0390d2163349274b889 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-45e3933b | sha256-dcc644c4a214f22353ce41a429072368e809613a6609cbec690f95a4a4840893 |
