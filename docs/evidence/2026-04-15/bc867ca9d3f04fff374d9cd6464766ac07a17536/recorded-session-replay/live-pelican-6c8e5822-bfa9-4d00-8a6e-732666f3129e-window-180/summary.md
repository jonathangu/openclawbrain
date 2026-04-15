# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-180`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e44bd492f128c06a27f22e67cd820199254d2a3ed0a6ac13485df4261f57fa9b`
- fixture hash: `sha256-cffeca9e647d7d047b9dbfa0c2bd2eddc1a7b9897467d5e861f95728aa0ee6bc`
- score hash: `sha256-8cecbe79fb916b276533a50d5bcb04c2351d15f2400e1c354a2bdd71144f543b`
- bundle hash: `sha256-5e4e4eb538613e687b94a49b1922762dfae5555c7f4ee83d200db709d29ebb5b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7211fa1ebf40e19b79ecf69c6d2f4cdaac759ca9e3451e680c32982ba6c5891c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a8764d8ab0b759b5210239f501a67b1e54b43e6079a919d5ba464c5aeacc280a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-57b78851efeeda92763965e79f978f541306fd574e56cf0002b1f6a8a847dd60 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-24a28159d3ee7c6cd55b6187d9950a2dc5feabb58a8fe853202c85b47461ab02 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0eac8150 | sha256-a52552831c5153dce4131f1eb3ce18ca8eb8e872a69b49f87f7bbd27929a8603 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0eac8150 | sha256-a2d956cf52e14c225f770a7f54bc43e7665630527d378521c5edebea4d1590c2 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-0eac8150 | sha256-a52552831c5153dce4131f1eb3ce18ca8eb8e872a69b49f87f7bbd27929a8603 |
