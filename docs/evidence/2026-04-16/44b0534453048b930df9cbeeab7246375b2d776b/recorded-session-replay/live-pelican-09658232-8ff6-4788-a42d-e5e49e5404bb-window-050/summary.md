# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-050`
- winner mode: `graph_prior_only`
- trace hash: `sha256-edca99fed5bfa02f8196eb002c7ca5449e3ad77a55a7ede4db613e0216b9a288`
- fixture hash: `sha256-075e6ae40abca95623d0eaab9386a47facdde038c6af2c88f5255ae9a6184b2a`
- score hash: `sha256-9fe1bae85babd15c3c6198e7968fa6231d8cb379dda56419fe3bbeb35a30811b`
- bundle hash: `sha256-3e305b73542b578da1081d87f732b78dc0191ef59f6f573b6932d36ebddba042`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-6022525cea9be81df298416f8768122250fedda0d93e17bc4857c9bee2c2bbc7 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-06c7a3cecdbb4d46bb10b442cd5fe444075e8dc5bf34896d897a2952997b5d10 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-6ba944e6844973025900586a34898b053d0a44b9c1d489d3863df5afbbeea362 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-23e44885eb8fe6697a5142a25831fb45947514bfd5486e7c7aca79cbffbd490a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-28f4cc3e | sha256-50b4585b235c968fe9dcf01cd4e02fdf6600d4dbcee6b5020ed0fcbb93d37a6c |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-28f4cc3e | sha256-50b4585b235c968fe9dcf01cd4e02fdf6600d4dbcee6b5020ed0fcbb93d37a6c |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-50afc125 | sha256-bee38a64222287921d9fcd6b52021bdde1cbd125de356482a4a7cd6add55c27d |
