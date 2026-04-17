# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-018`
- winner mode: `graph_prior_only`
- trace hash: `sha256-97d7d39c8ed80340fd41820d6d636bdacbec2fc0c19c6596d376217775b20481`
- fixture hash: `sha256-cee22d0c8692c9c54ea684f49e1d3ac5076518c4157aff7a2d52bb3e3278c63c`
- score hash: `sha256-4fe228463abe215143a1d360656dffacebb1322f27c9a006b077064eb6e93253`
- bundle hash: `sha256-a3efe17806c2517d5d2105bf0a91cbbcdb64b4ea7f89cef4c238f9a485fb34bf`

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
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-483ec85fd81a9827220f2a343c484a6bf92b951cf19018787d3fd3841e13006c |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6fded5194e2ddc419c04f0ff4433d7d583ad66c7847a159e5f81ad5680438980 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-b84f8c4abefb6cf800b1822f92b0d598282b3a1fca3e64452c04d92497a17a66 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-28ecb37a | sha256-75f1d517812659e8859b53ff81040aa78e3786ee4ac9ceecbf45930f751c438c |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-28ecb37a | sha256-67794f70811d09cadf34e98c72108a8a1ac7d0d25b1de1cd4a451f2b66d276a1 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-3d33febf | sha256-6aedb478aefbe9c56dbbafad87fdeed29867dbfb30fdd4fa42ae7032c085f725 |
