# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-020`
- winner mode: `graph_prior_only`
- trace hash: `sha256-970ba48dfa6c96d0a4965b4677af4fd629fef3cbc40e01188dbcdc91cce4557b`
- fixture hash: `sha256-be39fb4084ab4014f594ecf827b8324c7590b1b3c6ba2cabd9bff2dbd9a1798b`
- score hash: `sha256-0d5dd1f871ea40935abd903d1dbe88e9f7a86f4ab308a56969b37bd048edf63d`
- bundle hash: `sha256-c4e29d930d41e5e6e6b23826e991944b4d33e518889051a15359d13e2552f2c1`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-a8c8fb966bff98fd7248d900de12653a4c0149cb3145489937f87d5ed585d1fc |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-7e9d8bb4ef88ac4ad40908ef155c99a9b45e35412b6a3b5b850a61835be1a9ac |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-8ee14dfdd8112564dfca69db152c086e52f7ad3e719900c565383ea12e692d55 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-ba5d0d8e5b32078e4911ccc71a11a5fe87383dd635f9f0ce1c8cb21cdddff018 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-937029d7 | sha256-cb0c6d3350459a6efd572015bf6245c991610d54d09cfd39b59c78569f91d98f |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-937029d7 | sha256-9549459e1407d0b1b2a629ac1e06f72e8eb1dd3e6bcb47560848ed3a5509b50f |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-2ef6cda6 | sha256-64d39ac4c6f5591b6dbaea53cf33cb99a4fbc18bfb2d48f928a89cae9e99f5cf |
