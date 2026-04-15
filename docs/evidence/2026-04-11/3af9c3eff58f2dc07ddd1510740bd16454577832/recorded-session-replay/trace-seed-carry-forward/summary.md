# Recorded Session Replay Proof Bundle

- trace id: `trace-seed-carry-forward`
- winner mode: `graph_prior_only`
- trace hash: `sha256-21ed33015f7a51ad5fc95030cd8188dbc6655b191e53b59bec58141493db1904`
- fixture hash: `sha256-912c47fa6e90f710951f9473e5540907d6ccf58746703d4574c6d5fb9a0dd66b`
- score hash: `sha256-c149412e84dea6b6ecae1c977cf7441f71107737b032b00479c613e3a782366c`
- bundle hash: `sha256-fc5f1064e38f8325da75454cde6c36021a21289b08915de378e54582bded8524`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 6/8
- compile ok rate: 0.75
- phrase hits: 6/8
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 2 | 0 | 0 | 0 | 1 |
| vector_only | 2 | 1 | 1 | 0 | 1 |
| graph_prior_only | 2 | 1 | 1 | 0 | 1 |
| learned_route | 2 | 1 | 1 | 0.5 | 1 |

## Hardening Snapshot
- compile failures: 2/8
- compile failure rate: 0.25
- warnings: 0
- promotions: 1

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 2 | 0 | 2 | 2 |
| vector_only | 0 | 0 | 0 | 2 | 2 |
| graph_prior_only | 0 | 0 | 0 | 2 | 2 |
| learned_route | 0 | 0 | 1 | 2 | 2 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 2 | 0 | 0/2 | 0 | 0 | 2 | 1 | 0 | sha256-d38ca63ee2f4351248154268de4a796b6eacae65de64f0d136e785ddc845ea9b |
| vector_only | 2 | 2 | 2/2 | 0 | 0 | 2 | 1 | 0 | sha256-98bee8bd17fb8d4a2dffa93afbfe9cc7906f3f240542d3ce9cdcbac43d1d4338 |
| graph_prior_only | 2 | 2 | 2/2 | 0 | 0 | 2 | 1 | 0 | sha256-148ea762f870e578311c6514d2cda577dfaf1a5aa71345c25ebe309784b82612 |
| learned_route | 2 | 2 | 2/2 | 1 | 1 | 2 | 1 | 0 | sha256-898edb2d1e62f7f1edfdb532886132275840f643e25249ac0d9e83784e03d2fb |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| no_brain | turn-2 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-b3e86883 | sha256-7a222b5a883e72cd9e13da76a8005d7627af4364849e25b3d6b7c3e5084a0300 |
| vector_only | turn-2 | 100 | yes | 1/1 | no | no | pack-b3e86883 | sha256-31eb3741d65bbd4509ff8feaf23d55546017631da4d4d64ec90b286f58945133 |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | no | no | pack-b3e86883 | sha256-7a222b5a883e72cd9e13da76a8005d7627af4364849e25b3d6b7c3e5084a0300 |
| graph_prior_only | turn-2 | 100 | yes | 1/1 | no | no | pack-b3e86883 | sha256-31eb3741d65bbd4509ff8feaf23d55546017631da4d4d64ec90b286f58945133 |
| learned_route | turn-1 | 100 | yes | 1/1 | no | yes | pack-b3e86883 | sha256-7a222b5a883e72cd9e13da76a8005d7627af4364849e25b3d6b7c3e5084a0300 |
| learned_route | turn-2 | 100 | yes | 1/1 | yes | no | pack-6335373b | sha256-5791699e1c73553bb97dc72a7c607c9cd7c104b558b9c14fecdab3f177a365c1 |
