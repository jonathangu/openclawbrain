# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-048`
- winner mode: `vector_only`
- trace hash: `sha256-9c32a87b231e4d5848a772d9d1cb8d355e8b17c5c883fc0f1ca8776ef042ba2c`
- fixture hash: `sha256-66d4441e9cd89d5df06e129fcf70accf27e8123573950bf81a6f813e2979adc4`
- score hash: `sha256-fb1b34a1bfa043f721433c837e0508a4159b918ebd317f311cdff4b148126d18`
- bundle hash: `sha256-071c7400af1cbffb79b2c88a232a855a8fe17651cac1bf416696a9cde2da1f61`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | vector_only | 100 |
| 2 | graph_prior_only | 40 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 1/4
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2b9d9843197c6ea9cf1bbaf94c65647f4ecfa1e2224f8678711a552cc896cd7e |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-a70a35918dc367753736644f80f778bb4e82410cee6d92ef3391a622f2a89d22 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1829c56bdb1e6020f90099167d945e55c21918504a603cb828940add487d6490 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-3c7747f81af22232b13d7d65c69f78b0ca6ddcef8e2c038db30f37a3900815a5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-b99f0c6b | sha256-e2091f6e9ca5c60c529cb363ca6f0e942e444fb2db07a4c55d8af5363b535e99 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b99f0c6b | sha256-054f7b2ea81779f8ff3f6ed1536652fb81c3d4e8c351afb0b015f1483795efa3 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-08b9f50a | sha256-b0ca82b5a084ea684c4737fae59b349cc8d703a14afc6636372113bb6b62b0a3 |
