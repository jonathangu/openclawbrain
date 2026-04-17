# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-050`
- winner mode: `graph_prior_only`
- trace hash: `sha256-edca99fed5bfa02f8196eb002c7ca5449e3ad77a55a7ede4db613e0216b9a288`
- fixture hash: `sha256-075e6ae40abca95623d0eaab9386a47facdde038c6af2c88f5255ae9a6184b2a`
- score hash: `sha256-cf414bccca904e6b7d94017bc76ce6102a58fe30cb82048078663f48c88290f7`
- bundle hash: `sha256-5988cc7a08ddf02565e9e1b1b51384da46be036a83836774a017d747590d15a6`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-6022525cea9be81df298416f8768122250fedda0d93e17bc4857c9bee2c2bbc7 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-999240f8790d42ed9dd17ff358a025be1efbe349c04561e005a6be563b8d65ae |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-95a0edf4feef808356168c734ee2df0ba33353a62e539184337706b3fe5e40c7 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-96a815e41a7bc842531ad78efd5484c7a03262f0d0dec56553dc06e0b43a7876 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-5213ec47 | sha256-697d0fed35401eb8d43743fa0441b72cc6e77fee4759f827711f0205008bb30a |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-5213ec47 | sha256-697d0fed35401eb8d43743fa0441b72cc6e77fee4759f827711f0205008bb30a |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-79cee12e | sha256-5a7a0a4f1fdac63f069f175d3f863dc9bde3e3a9201c5e28b479796d2e68abc2 |
