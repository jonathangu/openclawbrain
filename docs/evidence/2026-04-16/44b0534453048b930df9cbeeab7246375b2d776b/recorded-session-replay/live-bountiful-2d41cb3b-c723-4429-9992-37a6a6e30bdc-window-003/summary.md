# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-2d41cb3b-c723-4429-9992-37a6a6e30bdc-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-580d4f8ccc5672e0994a6e33aa91736865fe4849bdf6f4307be6ada1929aaaa3`
- fixture hash: `sha256-f2131456430264646f8b93eefc85baa48a48ed730efbc3d47ac8e04c07a9e06b`
- score hash: `sha256-a76d4c440d7175a4e1052413c402c2476bdaf5d1a1937e3c593d3491b02f8e0f`
- bundle hash: `sha256-75a14ec5fbf1b53f8c68c3e14fabdfa616bed3ed9bd614f640b188cbe46662cb`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 70 |
| 2 | learned_route | 70 |
| 3 | vector_only | 70 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/8
- phrase hit rate: 0.375

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.5 | 0 | 1 |
| learned_route | 1 | 1 | 0.5 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f4cf4becbb2f48d6942aa18f5479385f6a9e60c52e2114c0c244447f1e9cb2ba |
| vector_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-9c09c360cc45819a25b7908a13f28e3c0c0c89399bd5cdef79a8d96644fc3f84 |
| graph_prior_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-50f2430f499cfde2e7e0492156ef73571d42c1aca3aca8b230b46ab69fbab522 |
| learned_route | 1 | 1 | 1/2 | 1 | 0 | 1 | 0 | 2 | sha256-e98ddccf44f4821573b8814d7dc236f3d78eb31737f35ff8557850a3963519e2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | no | no | pack-3d75b303 | sha256-d3905fd01e5689b039d850e2d38ccb40855ce8521af4879b622d8f163204a511 |
| graph_prior_only | turn-1 | 70 | yes | 1/2 | no | no | pack-3d75b303 | sha256-154ded79194f941fd5f3782fb6776035c3c784bf95dc620c6749f438f5bf4d14 |
| learned_route | turn-1 | 70 | yes | 1/2 | yes | no | pack-ca601306 | sha256-52f9dd1251799fe6fc23399e8f344babe4cb86111dfc59923b13cab1f42058a6 |
