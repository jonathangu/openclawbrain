# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2b0cf5e7b3f41bdf7d892185413608401c24e9c3ad252c16335ba4fe2f91cdd3`
- fixture hash: `sha256-ed8248c9b476e9fb2d02b9891cc8e11da35a8ba49c308ca9793fd2e0cd5daeaa`
- score hash: `sha256-764c1e9c2e6483476aa0b30eaed5433fedc05f63b20b4b5459e99d1a576b2cee`
- bundle hash: `sha256-524071a9d81ba6154d089006cc6fe04b6d5ab81ca6133571c4a833e9f169eb5b`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-368ba7ad0e0062707beb6bc226c2cae8531ed592ec4225d05a99c6ab4df81531 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-62490ac6472f24c53b0aaf3a65af4f8649fa4c3d633ad11645d58aa2155b6e32 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-0ae043f8df251d6877d6cee5d6c8a5f7774c5e3732308eaf03d59ea125014f69 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-3ec3d93f72d27cd62e94ad68188fbd8410f3d889be69775513170f098e3c782c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-857d1ceb | sha256-53977f3d16e5e773ea588c86d0b7c6f7fbf0e0b97df7cae2af9908006ce6f817 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-857d1ceb | sha256-a038142e445e87cd0260e2bc0dd30cbb354ea5cd7951d87341dfedef59e3f559 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-857d1ceb | sha256-53977f3d16e5e773ea588c86d0b7c6f7fbf0e0b97df7cae2af9908006ce6f817 |
