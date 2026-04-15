# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-077`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7fb0ffa9f4e8d70c4fc8ddccf35cb362423daf7116804c6734e40b9d0f4296bf`
- fixture hash: `sha256-810d932d8ac4f8eed98f074f82298ad7f5b0354d5fdf19533c533df6c21240d2`
- score hash: `sha256-f34b3fc61b23085f49c265e27077bef465510081426f33ea03f87a3ee353176b`
- bundle hash: `sha256-2f83fcaa27a1192a0874aaff574ce04f7ade0c2b6fe63f4f0277441d91dea6f3`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fe2cee15d245f859cb5315bfc802316abb7874a3bff97839e84f3440b5d4a896 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6ff700507370cc2088c05d6d4ab706b8b8b0b52803d5ee6f81cda0dd072fe1a1 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b0940762571ec87fa617f1fddd1caa077b3d6a817007f066ca0a65db3b74d9c2 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-3715db82801e20853f5f169922ef2e474a9c282dc9b094b044c37bf56cb7e7bc |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5d6ff1d4 | sha256-f741f0c0bd389f09d8e151e7c3790e27b369252318dfaa76bc50715f1726165e |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5d6ff1d4 | sha256-8be61016b2aab076a2fcd7b93e983e84ba823c973518939f4d68ba14c65298ad |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-5d6ff1d4 | sha256-f741f0c0bd389f09d8e151e7c3790e27b369252318dfaa76bc50715f1726165e |
