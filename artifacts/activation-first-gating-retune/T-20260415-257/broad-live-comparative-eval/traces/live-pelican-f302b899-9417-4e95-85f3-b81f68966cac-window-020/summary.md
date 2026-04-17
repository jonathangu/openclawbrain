# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-020`
- winner mode: `graph_prior_only`
- trace hash: `sha256-020e5fa0ec60c9180b8ca12d4a8cde03c3eaf93efdc6e1249456178218366170`
- fixture hash: `sha256-2fab851b07744bef46921e5dde6e3c44cc707f0e47e7a2b971ff5ea69c88de53`
- score hash: `sha256-f0cabcdecfd8d53e6832fd5b634186019aca97b8aa99792be5c9ac724cca6f30`
- bundle hash: `sha256-ba968d93d0ba26a2d9feab1a8a895e69bc851dd3d73e45b562202a179c07f615`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d84e9ddc31e34697064a9e60de43374da82ef3d65551bc6676137ee0e90f5d63 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-589d1907ebd6756d08ce9e51499a320ac4470821032fa2566861c5c46441a5ff |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-31965e081c99e204f0890242d72e042873b0c1bd5c33daadfde7102fe0c8878a |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-2965f7c4ab6d482336c77c617c2e69e3ec3b20445317abb1a17e188de9bea8d9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2ede11dd | sha256-6773f9c6bb4a928101e2883cedeee548d5f68a12ff0bc3c9262d5bf4d01065bb |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2ede11dd | sha256-b6a2aa532a9f0134f025eb07e23dbf2425fa2e52e6b3706b714cf9b83e43d8a6 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-ca1fc6ae | sha256-37e435bfd121097fd3cbb0bcdacf73df67df09e6f337f91e21eb7610ea513864 |
