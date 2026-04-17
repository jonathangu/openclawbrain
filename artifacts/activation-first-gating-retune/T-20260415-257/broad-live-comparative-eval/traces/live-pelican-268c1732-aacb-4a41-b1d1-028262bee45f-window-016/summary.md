# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-016`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6bbdcea0bcebe80c4396f90ecefc21d80712c695057e642f25e10243f87c4c7f`
- fixture hash: `sha256-e1e5a273f33109e97564303fef433c5ce8b0488cb943d73553c438e3bc4b82f9`
- score hash: `sha256-d474ef36f06041ff2baa75297ef3dc7fa3058b5d96903995323e26bb9727c1d2`
- bundle hash: `sha256-c1f85f45f0cbaecb4fb7894226fcbf00f2d468ff4f5f5a7b32c14d2a3c447797`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d7d1f1cbb97b76e814b60c9940bdeb937cd1496e6d58b8c2941e362dd90a4031 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-72a2834db928aa0755951144130c5c98e34b9adc6fd9fdc8d04073885ca9f79d |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1479388ba2e75bc0524fb54b3e116dde94b4bda26fa85282a26376658dec1367 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-e036ead2809cfbd6a076069015e1fe98f7547bbcfed2953f69b084ea088a94a8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ac1ed9ed | sha256-2e139169b0437259fc182f1b611fca30a2f38792809ea30f3a2b8b842c5af807 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ac1ed9ed | sha256-faddd5ae4f641a8a2d6a71fdc70348f8c93704b4040bde4b1b67d94c120c9b03 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-76890aca | sha256-4eff359260ba253322b5ec1249ae37cb36084b48d37fa712ce52d4de3192ec48 |
