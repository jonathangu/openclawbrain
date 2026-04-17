# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-016`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c817aa28a6ea88ab750b90d075966003c4144ca68cee4de31510afc8940af725`
- fixture hash: `sha256-12c8924300be23df2d629cf06b8bf4e9466d47a9b90ef4b0770c780fb827282c`
- score hash: `sha256-27105c8ab07c564e11fc957227d6477cbec8a707d8b437bedc16667f4e1c40f8`
- bundle hash: `sha256-5e87efffc128155b22364b5677701aad3724d795d9654b1b190d70ce90a37923`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a755e36be001d38e08b764d65e8f6dd1b01494428975ffb22d7f3f721a73e79b |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-55f4d164751d65d783d061114271225caa5b61f0cce767c58835ff3dd646968c |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0da2764abdde94097f33208cddd8d21af7b383bd681273de33715674d231df0d |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-ddad01a7a952cbb96c740c672dfc252a901efd3a174d26e03e402072f5db7c3d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-03e8a1db | sha256-74feb6201731167a13c500af4b978571a05b644493ff9377c1bfc417e474964c |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-03e8a1db | sha256-4192c4d2533e384b5988a80989bb2f26b60e1b42a8db5e281ae8bdf5126cfc68 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-696d44d2 | sha256-7c876ad4ca1bccf3f4ddf386965f0866ae2b33f714159cd47afd5dd53f21f043 |
