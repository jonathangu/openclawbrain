# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-020`
- winner mode: `graph_prior_only`
- trace hash: `sha256-020e5fa0ec60c9180b8ca12d4a8cde03c3eaf93efdc6e1249456178218366170`
- fixture hash: `sha256-2fab851b07744bef46921e5dde6e3c44cc707f0e47e7a2b971ff5ea69c88de53`
- score hash: `sha256-8a8ceae17fca9016280cbd45ca26b2ef61573e5561e74f12f379019a7ae9360c`
- bundle hash: `sha256-dcb2ae7e412937f48a8e5a29faf1b1f54e0af69f68833d2768436d0b2112ee92`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d84e9ddc31e34697064a9e60de43374da82ef3d65551bc6676137ee0e90f5d63 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3dd9911fb11e579f9c4c5ab073e3fe0498d886892e09e3db7b9b138161a325e1 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-208e372e078b36bbe9f06ce3644a30bc8ca28acd06b41a857cbe0fc91541255c |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-ae509c3f98cbb5eebdeea961a55b1eaa946ea119a580a325358b3d5ba0a89dde |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-fcc95732 | sha256-8a21522e6c62b6db27aaf8a18d474d0f7a5784a85551d531b2e2a2e5df557ce8 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-fcc95732 | sha256-7e4267aa8a8d7e095867624e80d69f04fb9ff9afe0a897d388b1443638cf13a4 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-980b0c03 | sha256-fd5143858c98d571ddbb5a198b7c0d8d98d88fb6f01c80ccc65d3ee9a85f73ec |
