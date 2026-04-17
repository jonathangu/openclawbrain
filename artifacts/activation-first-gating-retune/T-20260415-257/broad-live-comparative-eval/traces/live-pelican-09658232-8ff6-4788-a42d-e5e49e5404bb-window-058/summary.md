# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-058`
- winner mode: `graph_prior_only`
- trace hash: `sha256-70e107aa90463a0c77bc30d344eca5153707641920ed24320747bbd52e05a0e6`
- fixture hash: `sha256-2a5cd5afc4b09fa9beced059043152cd23fab3958640aae8275a1e91138ba120`
- score hash: `sha256-1cf8232f599c2a48e0ce6e3bf9fc76ee67a968784a0b00bbcf58d96fb1df869d`
- bundle hash: `sha256-bb9cbf9fd8d7132797ab9a1c8d2a2ff12a73456c065373bd3bab78ee6f32fcd4`

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
- phrase hits: 0/4
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1c146c3106aa3a476acb28a8b075ba9caa0dc741d245d11ea00bbf3c4bbed6c9 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-19ab6424a774a5698860979b1f22d34dd29fca73e4926dc2bffeb738b25628c3 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4761d59ca7337e78f9f61f345d74ec4e4f966192daf89977dc0c7ee57a7292b5 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-d3583393a041fc04c4ce81676d2b3602504c9fae067f4f2aabc5136644f2b717 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-6fd8c339 | sha256-afa09d0dfa6c4383848af9c528202e1d6127dc0863534e4b663d025f3e5503d5 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-6fd8c339 | sha256-f49916ed3896704b258e1bfbd10228614ef1ac6b4516ac325ad7c445411c9572 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-35a4f97c | sha256-33bec7a1ce301a46ef82770bf188992d8d4d031b52fc566d8e7b85a33bcff839 |
