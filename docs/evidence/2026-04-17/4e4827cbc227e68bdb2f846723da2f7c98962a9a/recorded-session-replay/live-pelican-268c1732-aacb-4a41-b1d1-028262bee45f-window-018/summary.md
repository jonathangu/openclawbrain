# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-018`
- winner mode: `graph_prior_only`
- trace hash: `sha256-97d7d39c8ed80340fd41820d6d636bdacbec2fc0c19c6596d376217775b20481`
- fixture hash: `sha256-cee22d0c8692c9c54ea684f49e1d3ac5076518c4157aff7a2d52bb3e3278c63c`
- score hash: `sha256-5a8c74b47c06ee352cedc2780b6baf95d5e64923b27fb1ae0e7c993046332165`
- bundle hash: `sha256-2c305319e5c78c579fc7fbbed270e13e1171607d0d24bc84d1ef9c316fb71d27`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-37de16724e3f909b52770a9de834272378dcc6d8dc93db3d2e32057318f060c6 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-23bf4705db95f5e3ab6f661e05b1142afef45fb7ad1f57b13bedd5a8e0d2eb16 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3ca0e9215f76f3e483a512ab52dfe441a725ae0c796956e98df037b5b5353ef7 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-82fdce2e9612ef01848ce343fc1dd4480bf75bf0df52e2419872db207cf57bbe |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-51056591 | sha256-8a850f9ab66e6afc7f51066bd7df42967c5b824476a151998b2be366a22cfc3c |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-51056591 | sha256-25ea7f17459de27951cfd2aad8ed644379003d4888f066b118d7d07f8de19a56 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-654cb0d6 | sha256-6c66a9db8dc095f30e7e4c3434e92aa98e7e542ec8e6e04a4cd1855f8a5a5d13 |
