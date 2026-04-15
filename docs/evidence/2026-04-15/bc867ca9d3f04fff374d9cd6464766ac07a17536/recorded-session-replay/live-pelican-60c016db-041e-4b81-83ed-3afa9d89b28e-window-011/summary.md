# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-011`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f8ed2037bc2ff0feca7432af422fbb2b58a869f3969fbcd41ad42699329bf723`
- fixture hash: `sha256-b54fc3ca6fe17a912f89c0806fd3df709e1f6f80264d7323adb898abecd00677`
- score hash: `sha256-1b7002bfa6df5be6181883f5870c64d72f2485b06e7264359365fb3383d617eb`
- bundle hash: `sha256-1b39cd01ab6bb9ea977b7e8757b9e52dbbc1bb36f478befbc8690cc9d3df53ad`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-34738adcfd10c341d46efa61990a3844e1795a8fb18b5ebdc9694342a06a5142 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-207f62e72e5e814069f7d634ea4d40e459983a4817f29db82c583a8a70e90acb |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-e7757b5a600147c52ac895def90975fe14e9fed1d2c2f00461ddcbf0a1f3d4b8 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-ad1fab531ad56ac9be3d2cd9410294bcd6b45f3a6321bfc165d39038d234fce2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-26b82ea8 | sha256-096343d0ae0734c6acb9db4fe5e9fede045872fd0e72190c3b00b2d12bb49a51 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-26b82ea8 | sha256-eb1609017f3c6d42ec2e420aab0e54258d0da09f17734b8a937d04f5a6b3907d |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-26b82ea8 | sha256-096343d0ae0734c6acb9db4fe5e9fede045872fd0e72190c3b00b2d12bb49a51 |
