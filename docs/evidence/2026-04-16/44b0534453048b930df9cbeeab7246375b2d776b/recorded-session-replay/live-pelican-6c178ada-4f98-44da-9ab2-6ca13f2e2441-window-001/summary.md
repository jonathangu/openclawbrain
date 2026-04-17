# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c178ada-4f98-44da-9ab2-6ca13f2e2441-window-001`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5c1146574706ec395e6f5011fe3bdf3e510b31ef69670a55eff27bc156061d1f`
- fixture hash: `sha256-b0b87869202da9099b109d7a7b86f16484e8b3960b663b22dcb9b0c0fd925784`
- score hash: `sha256-d1bfdff62f6dfc6e44383dddca42a7ddbc82c38dac38f52e164ef4658c08633c`
- bundle hash: `sha256-7dde72e86dc4a007a08efa4847f643843cb1e36377b58497bad9d664dc0e32fc`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4ee8a7a4748f6af35d73940b990960d0c8506d722d1756ec1464f9fd52079a6e |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-59b172b6be5105cf216c40e77c18c39dfb90d62a556b607cc388509fa91257d6 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-75a71d93edcefa9393b0e040e6ea6345cef65d3d73a42db059aeaf25fe428c25 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-446fbc39b5b67f09d5ad59fd6e935faa5fbc10287268c77f03589ba3e3b4f88b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-fc73f00a | sha256-6f8067e4ed2625cef892fd0b0319d9853fc8a1c4d092157e3ae222b359e23b39 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-fc73f00a | sha256-c0ddca812b3a19659b428c927139c71710759d287458bcde069bb927581347be |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-7c21d57d | sha256-4ac4c81871de1226141fbdccd62b5c8058b74dbf770cf7c488d65335d145f874 |
