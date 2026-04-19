# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1fa535bc6d12bb71f464c3299ac8be0aea3ff43ab51a534634c743c27f7f228f`
- fixture hash: `sha256-3d8b02b35686d528b31efaf88c8dfba1c6ab218b6a7b654b6357f35e5f82ffce`
- score hash: `sha256-3c9cf15df57f95431442f4783c78393065ad9bf73a50b99f63db34a603848735`
- bundle hash: `sha256-2ec109c91af85b362df0f5e4b8712280b10e6994f35183b3694d4606834efba2`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-94d47cd92404766dbef910ada1cf5f255eef0ca70682ef6fa7b016001adffcb3 |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-df2f34d5faf6a446420aad586702bd4514ce3ab62d42c5eeb27a5e0e06089eba |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-5430bb8524987d7002589be58d270ef60f9c1a29427f62833db67d4f38597baa |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-6044903a85d1401471842a930c089505db2900b48e3748ebdb70a18ef3db2297 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-044a6800 | sha256-5b0eb8edba4622df61453c4cb07ac479aad7811a1a2b258181c84ad6e33265a8 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-044a6800 | sha256-c71e603346c705982744f06813b9a351578c77d7c136cab1ccbfe3f058f10690 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-044a6800 | sha256-5b0eb8edba4622df61453c4cb07ac479aad7811a1a2b258181c84ad6e33265a8 |
