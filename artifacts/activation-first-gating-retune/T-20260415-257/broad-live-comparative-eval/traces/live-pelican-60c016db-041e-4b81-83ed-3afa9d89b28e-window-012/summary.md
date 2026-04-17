# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-37ca78c4f79af1f5ccebd457f09d9bd9f0b270ac0d1c7dc3ef10aba20d199a04`
- fixture hash: `sha256-221b36f5e3c3b83dde39237b8133ec3e68acdd74bce0b4e3672a3fac84a8cce9`
- score hash: `sha256-65a4eb91c0e3ec546350a167078ee1adc23b4c895f9aecf54ae1e22b74065dc5`
- bundle hash: `sha256-653995fa5e0df81d4c3d209813d35eeb415793511453984b0eeceb2794ea63c8`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fa6a30b0b756b7163e1ab0f1526218df1fd81b134bd908830d7627bb5155f717 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-bffcf5dd909abbd18929ead6324494958471f4d8fd7f71475b438aca9223bd8d |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a92e9dcdc32b99416aeee0e4bc9f100b4d8d161b590c4a2afe4504ce613c1d60 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-ed3fc52682fa6aec6710fa3616f843156a2263cd61d16b6db11009105cf35efd |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-45acf0ae | sha256-479b373644931138f2a969ae2f153bc47109c41eea60d825365a53ac7c2c5db2 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-45acf0ae | sha256-5087dd7f28e80c658225715db441f8b780e4b8481d71f27b5e372d187f668149 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-39e988f1 | sha256-62ee23e78e58120836bd58962e2d796972c0fa5ad728cea0b1ae5a53b5e1e025 |
