# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-016`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6bbdcea0bcebe80c4396f90ecefc21d80712c695057e642f25e10243f87c4c7f`
- fixture hash: `sha256-e1e5a273f33109e97564303fef433c5ce8b0488cb943d73553c438e3bc4b82f9`
- score hash: `sha256-bc07e4e9cf2032094d4287377e7766adebb2cdf40e33d196c2bf01bf0b82904e`
- bundle hash: `sha256-1632c5dfdc8e9a3f2e4f6a20e734ed3c1d76a44255953f6053360f8316421d01`

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
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-88b07e61203bafb86de24750fc363936197a2d4cb3d76d09a5406cdcc4f72012 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1ef72231f5ef627b0fe5beb8e10baa943acc57cd16545da9e1e5890b0df08e45 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-0b5ddb9e7ec64ce45e9e2211dd40d2e5f2a05bb95769ac4fd676c384f5fca556 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-34ce9463 | sha256-71ce63599dccbb736d70b370c1865f95f3abd2da2fd22beac7f1ff49293adcb0 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-34ce9463 | sha256-63a20e7f3dce6a8987df07e8961f741dfb675829986c443593d398c3268d1a7b |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-34ce9463 | sha256-71ce63599dccbb736d70b370c1865f95f3abd2da2fd22beac7f1ff49293adcb0 |
