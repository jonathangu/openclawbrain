# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-020`
- winner mode: `graph_prior_only`
- trace hash: `sha256-970ba48dfa6c96d0a4965b4677af4fd629fef3cbc40e01188dbcdc91cce4557b`
- fixture hash: `sha256-be39fb4084ab4014f594ecf827b8324c7590b1b3c6ba2cabd9bff2dbd9a1798b`
- score hash: `sha256-eed3f1c5b82fd1cc38d3c1fa46dd757e0a5c4884e16c46aeabe36e11ff6ffb98`
- bundle hash: `sha256-14de9424fb0827493d5a55285204fbb0f826e60b5d963f43e7f88cd987623754`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-a8c8fb966bff98fd7248d900de12653a4c0149cb3145489937f87d5ed585d1fc |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-35431fcfed3ef5d4130fdfce3c451e8f30d04a7b72e6a1ae5bd46a4b46b3e9b9 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-4901e60a2d7aaf761c1468c48d8bde796bca067714aeb230a0730bbf9001fff7 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-daa6c510305292f5f59c1f83aa1bff06823f81922d74a6e1f5b45d18cc07476a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-9b897750 | sha256-89435db08b4f16cc620662562180a95f3f7feea5d395582d6a71b31d39d7a5af |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-9b897750 | sha256-86eac7e2aa613ff1a0e269a0f7dd345538dc575c7c64a6a00ba58a2e89678b1e |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-37101b1f | sha256-b69afbc90a2401802a7388993812528257987eb08691cdcc4284a9c1c6a9194c |
