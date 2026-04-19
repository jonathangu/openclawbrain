# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-067`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c03db6636813d15ab314e1036640845c998fa263e7aac887b43bd18e611eb255`
- fixture hash: `sha256-a5250b921f3f7515a2c7a2c53cad821f0baf9317cf0101574260207ef30568c4`
- score hash: `sha256-4871ffbe318c7ad4a57d1df77e34d8ed0bd44d8626b5d3123d39ac8f27c372f2`
- bundle hash: `sha256-572aba73e9053af5bf34f122e6ccaf060cdfa9346bd0cb6c4a8cfe94ba99f587`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fda1d2fc29e8cba170ae68633dbee693f7e6102e2f256c7ffe0a2134e709581a |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-6e6092bfdc33092f2a12af9d8e6c1cf078cb24a836a59ef81f575e3a03d5ac80 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-ddb6b3fb17bb63bef4371f12603a6cfbacc1ec614c701c17a318214ea1cef55f |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-1642096296e2e58c229832aafacb6a997376e0643e0d6d2f17bb55a5c5472e5a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-4c921c2d | sha256-00659b7b16f2c5e819a544d042245b3286036fa5ddb105f22490323300be5f6e |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-4c921c2d | sha256-ea4b54c4275a40d8883ac728433e8bb632951e23a6448f3016d28bfd77eb2c24 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-4c921c2d | sha256-00659b7b16f2c5e819a544d042245b3286036fa5ddb105f22490323300be5f6e |
