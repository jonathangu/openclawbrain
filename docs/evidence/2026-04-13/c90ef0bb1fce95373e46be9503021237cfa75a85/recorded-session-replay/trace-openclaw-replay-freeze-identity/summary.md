# Recorded Session Replay Proof Bundle

- trace id: `trace-openclaw-replay-freeze-identity`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f69fc254681ee124b61d343539bc571afb510dbef06d8cc553fca8f4a3781603`
- fixture hash: `sha256-7a70449faa887e7ec02d5a8115792adc662e98498339bea4d10387f0ea078086`
- score hash: `sha256-b20e541eacdeb72c9b549f163911cac3c1a9dc5ffc39dc7cb8f2a00fdb9e100a`
- bundle hash: `sha256-1bb21b8fb62b87dc1275d0bd4d3a0f270f1fb6bbb443f57bd44c5637354a3bbf`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 9/12
- compile ok rate: 0.75
- phrase hits: 9/12
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 3 | 0 | 0 | 0 | 1 |
| vector_only | 3 | 1 | 1 | 0 | 1 |
| graph_prior_only | 3 | 1 | 1 | 0 | 1 |
| learned_route | 3 | 1 | 1 | 0.666667 | 1 |

## Hardening Snapshot
- compile failures: 3/12
- compile failure rate: 0.25
- warnings: 0
- promotions: 1

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 3 | 0 | 3 | 3 |
| vector_only | 0 | 0 | 0 | 3 | 3 |
| graph_prior_only | 0 | 0 | 0 | 3 | 3 |
| learned_route | 0 | 0 | 1 | 3 | 3 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 3 | 0 | 0/3 | 0 | 0 | 3 | 1 | 0 | sha256-d0d7ec5bc630c9366bf082fe9e3df18c7cc3b50bc509aba5c3b112c7f54804ea |
| vector_only | 3 | 3 | 3/3 | 0 | 0 | 3 | 1 | 0 | sha256-cd2a17162c2567c4600de98468f04595a7c465d1d698ed3692f0425230c7f2a0 |
| graph_prior_only | 3 | 3 | 3/3 | 0 | 0 | 3 | 1 | 0 | sha256-0e68afbd0224ebf64f83727cecf4092a8ad0890441d7c1a5e5720db9a184f321 |
| learned_route | 3 | 3 | 3/3 | 2 | 1 | 3 | 1 | 0 | sha256-9456347adfd7c0498055a80d55ccc5816d8cf831a79cfb397e9dd7817f36a599 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| no_brain | turn-2 | 0 | no | 0/1 | no | no | none | none |
| no_brain | turn-3 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-feba6168 | sha256-63b66a9676b9754721badfc10593da2c3b30b5647bb4b30e2ca07ec9f0eb5728 |
| vector_only | turn-2 | 100 | yes | 1/1 | no | no | pack-feba6168 | sha256-04f230df9652dc6908acb8f9fc7dc513d0a30f023dfc0834db3617a648100610 |
| vector_only | turn-3 | 100 | yes | 1/1 | no | no | pack-feba6168 | sha256-63b66a9676b9754721badfc10593da2c3b30b5647bb4b30e2ca07ec9f0eb5728 |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | no | no | pack-feba6168 | sha256-63b66a9676b9754721badfc10593da2c3b30b5647bb4b30e2ca07ec9f0eb5728 |
| graph_prior_only | turn-2 | 100 | yes | 1/1 | no | no | pack-feba6168 | sha256-04f230df9652dc6908acb8f9fc7dc513d0a30f023dfc0834db3617a648100610 |
| graph_prior_only | turn-3 | 100 | yes | 1/1 | no | no | pack-feba6168 | sha256-63b66a9676b9754721badfc10593da2c3b30b5647bb4b30e2ca07ec9f0eb5728 |
| learned_route | turn-1 | 100 | yes | 1/1 | no | yes | pack-feba6168 | sha256-63b66a9676b9754721badfc10593da2c3b30b5647bb4b30e2ca07ec9f0eb5728 |
| learned_route | turn-2 | 100 | yes | 1/1 | yes | no | pack-9380a825 | sha256-fa4608a1d25a420bedb7103692a4208af11aa761995037ba3b55cfc4c0e4b40c |
| learned_route | turn-3 | 100 | yes | 1/1 | yes | no | pack-9380a825 | sha256-3b8d309e75d2b2fe9396fe3257ef5445a1d2ee744eff4e26324acb8bec8010e7 |
