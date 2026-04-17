# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-056`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3b194659082568a82c511a7152ca31a2b1b95a8940775e0d8501ad2641699262`
- fixture hash: `sha256-818172b532c3157150cdaf4f843fa921402c9f435a9b49f1a0bba05b616c0656`
- score hash: `sha256-19bec722be8820203022dc7b531b2328b971fe077ac804821e9820526372867a`
- bundle hash: `sha256-d56f919f62cbf212e12d8b1d72c39f9f8a986ec7a34d6ab971eef2863d5f8ddd`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-3c006aa8496cda3a74dc0aceaf43d36eb374dd8330caeef238bfc730df80da87 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-0289a3f87a2164e3da93c34e1601a8a3e964da30c6d63202692a789533aa5344 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-c02a096ba6e19f643e84cfa743ea29dc571e2b8bc03148c91b936f776ca8ba6a |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-1c6daa022adc11fbfdd55248947d5e9f426fa8c55301980b7f613d5f4a61ebba |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-55b30bad | sha256-f9b31ed634638ffa0ca46b9b6c69a576347bdbf8d36a35d8c0bac9a44d70da2e |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-55b30bad | sha256-fcb2844e36041c51eee0fe838de92d34905e99f1c98b2dd38c498078dbbc0333 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-a0e36a36 | sha256-c0e5f7425c46450d9edd010c5e9a5057c351b90f27d2c435417e2ca708de04b2 |
