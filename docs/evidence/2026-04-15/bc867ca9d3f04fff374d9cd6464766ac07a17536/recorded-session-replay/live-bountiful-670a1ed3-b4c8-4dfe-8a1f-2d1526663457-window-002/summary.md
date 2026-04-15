# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-df424694932b0793aaedff791f54d5ac971c24ed551452ee216f10c505396c8d`
- fixture hash: `sha256-cdd5cd85fb616c8f44b236f115a79978bc2dcad4597a177039207ba517f1bddf`
- score hash: `sha256-28a34dc31c4086d7c5640e73bf34a980bbed1a6da84de4b4c5abdf7f5c3e159f`
- bundle hash: `sha256-85b1d9cc135371ba1cf1637bc1dfb27a0234f8b8a0cc951fdb7ba89670192d6a`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-df3745ac4e10090248775f0174e4f7f9517bcadad1b8588a0276c1d2f867a57c |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2497220480193abc4e8297cfe2f1a21ddbac239062d315670f82ad1877626141 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b52f167b8858a972a7f231f41a0cdaa91c010cebe136db8979f808fad96309ce |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-33e09829f5cff344b610d2ccf0fda0254c776657e917dfaf0cc86634b15b20e9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ac7b8ed0 | sha256-d23d5876cca16ac4dbce18a965d3775ec0e02713189465522212bde261df2bdf |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ac7b8ed0 | sha256-d23d5876cca16ac4dbce18a965d3775ec0e02713189465522212bde261df2bdf |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-ac7b8ed0 | sha256-d23d5876cca16ac4dbce18a965d3775ec0e02713189465522212bde261df2bdf |
