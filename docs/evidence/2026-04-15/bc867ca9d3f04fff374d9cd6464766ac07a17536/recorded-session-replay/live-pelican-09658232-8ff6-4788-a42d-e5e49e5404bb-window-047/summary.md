# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-047`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1d8011faf3d69f32bcf0e92bcef735c94f96aebd8322b667cbe52a25917f1a6e`
- fixture hash: `sha256-f28ec0241ac4efd4c1f97733d381efba161e2d4c7cd778ddce2f415ed4529529`
- score hash: `sha256-00edffd5d386d2073e0f3e67598dc4a3744fe743fbdfc9fbadd96284c9143939`
- bundle hash: `sha256-cd4333b2e742611ea05d5a87f5971f097adc3525e6a5a72d1d433d7c845b9e65`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-64f932981a8a1428e017d3b3bf8eed9c04a8f1b43e3be668df16d36de77d3b6f |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4f4800fd709ad3a4def71bd5c7cd2f4cd553593634eb3d3ff7bf8bb980ae869d |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-de3156b87f243f8aeb41c46c67e6b15d9a591eaa72fd478f2da2412636d1dd01 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-5ce713f8b71caf49bcf9d0e75f649a8aea81925ee3f9ac460d8805e11e1bfa89 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b57e91ac | sha256-6b5fbe67e42461e4c81e58ab19d71692b7d9d24912c2847f2f570d38e1477348 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b57e91ac | sha256-b8f8f690cf431643ed38677f11798a1f46dc9aa6176082c7f0868deae42e2d51 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-b57e91ac | sha256-6b5fbe67e42461e4c81e58ab19d71692b7d9d24912c2847f2f570d38e1477348 |
