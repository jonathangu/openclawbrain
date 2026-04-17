# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-161`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2b83c4e3036cd20627db9b3b691867f0c4cf67798691db96c05537d9efc454f0`
- fixture hash: `sha256-d883ca17da8d181a1200f08513acd619f27d5b75e1c49c4953044231381c83cd`
- score hash: `sha256-c034ead7195708133da27269b4379db920c09fbeeb370052b46ebb80ee0b4f5c`
- bundle hash: `sha256-7324d952842afadc96b59acaf4eafaac565ebb6372e912fa7a509cb09ef8618a`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d5551839106976537fe1c9ce0dbda883b66824b3f67f3049bc2763f475be1647 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c16878e6fe57d1e77b1fe9016207a598a5ed63235ce589a8cc425b32cf39fb64 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-162a8e6036e2b48812f66ce19dec75fe402aa9da0bdec8dab1a3ec75750124a6 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-b995ca09977a0a892d351bff0978b231f674027aea5fd0150a1cf4ad41f10099 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-58e77ca1 | sha256-cf9ed88f614a261e72ccc2b2e900520d66a34616a78ae8147ef756b6781bc9bf |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-58e77ca1 | sha256-09ce6e44340f16450c5175eeb75f262fcb12d601d29382e6e486dfad5ea1c9de |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-6a9b1ffa | sha256-fa35096ef620aca967ca610edc1811333acf122213272fe6e85d4d83c65bceeb |
