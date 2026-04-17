# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-017`
- winner mode: `graph_prior_only`
- trace hash: `sha256-41fe4ec7878578e538ea87217d0f1ff26ff7c1c5009495fddc6dab6258a2bbbb`
- fixture hash: `sha256-49ec3fe73495575ef4a5edbb2b2c58d86b67a86df5a7ca6830045265a7717b0f`
- score hash: `sha256-9fcdde23a8bc264b54e0bffe2198ebf752f405980a99bfc6ca2243231d5de688`
- bundle hash: `sha256-0cc67514f8d0cbf9242b159b20acf38eb4b3b5aa757be7fbce03a1fa48cc5d75`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d5a96d1404eefca9da775b3bd1f6864e8e794d06c6c90d16aa5e90455db3aa7d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a36be4d2381e9e6f93c5825d936d2e4edfa4a3bc2ab9d8fe40abb72268cb001a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7e860f7d880b0a575b3a25e96df86eb9535d344b5e59469772c2b9ae0f6a0fa3 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-a016682cf9808838daedda59880ccafb9c944fb66e35558196f7df7dbed56c65 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8c2a46f1 | sha256-5e2590413c6f14fbbe382ff81b169d6225964c3aafcad2f6eb6280d03f3c3529 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8c2a46f1 | sha256-965a598ff4590dc7b42886b20982cda1c76107fe02f570af2df2160eeffc4a07 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-cc1fa206 | sha256-6cfe393d8efe9ffce7f176fd33bb0f33790a8cb14b1355b5eac29db4344284fd |
