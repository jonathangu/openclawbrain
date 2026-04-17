# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-161`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2b83c4e3036cd20627db9b3b691867f0c4cf67798691db96c05537d9efc454f0`
- fixture hash: `sha256-d883ca17da8d181a1200f08513acd619f27d5b75e1c49c4953044231381c83cd`
- score hash: `sha256-950a75e71b04f8f228e87e9d9ebfbe0396f55cc07c6cd0900254ec01805afde4`
- bundle hash: `sha256-7b7e71b1406b7f29838471d28595a6f4c1fd7edcb5b54d0404c98a2d6d776c4c`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d5551839106976537fe1c9ce0dbda883b66824b3f67f3049bc2763f475be1647 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-87fa9c4ee4df03792177a995d1f57b7bc385fe6964c4cca2a09a489c2b981e1e |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-16835fd9c2dc18319940adeceeefcf49788a7103eea9ca9e575681728de80fb1 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-88259200b45074ff0039c3a0ff4f3c4c44dece2c8b2a0cc4c148419a5524ca77 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-78e6e13f | sha256-cb964089ef3c381ea23773527478d429763019ebbaefb2d029e605619ef11e98 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-78e6e13f | sha256-1a43d29b442a9901c680c78c2bdc190303750878ff402634761eacb4d717771b |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-8a9a8498 | sha256-c0781548268c1ea030263486955c1dced81c15f7c5273e979256c0d430adef16 |
