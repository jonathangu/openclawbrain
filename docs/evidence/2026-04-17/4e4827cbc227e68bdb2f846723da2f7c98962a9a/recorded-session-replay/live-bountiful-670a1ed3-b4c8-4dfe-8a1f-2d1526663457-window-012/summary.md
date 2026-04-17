# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-14ade459ea986baa6e4e71bbbde0e89dc1fae7980400ac765d36815dff4c4f35`
- fixture hash: `sha256-9c30c978d165bf9a25e14aa9b77d9a12a45f7a9014b4a8204bd05ec1ae139d4a`
- score hash: `sha256-d6fe992ce25ef486f0bff392e35ab290768d733e3eab2c51dc3da54fbf969f64`
- bundle hash: `sha256-8d392c6250252b320cf4cdc5ef26f5674d74845341191ef3d7a118486b1acc98`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-334c135bfa30ec156738872f694abf9297995f829f0e8e1c5041f315be0a98b4 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9d7afdd5e83a80cfac78c4283bb51b4a53acf1af35c197377827af20752f6bf8 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ce48d44fbbd00d44fb879a8870b3a5e8b42e2c3d98d3c0bc8ddd192bc7a1ab43 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-9bf9df349c456b35480e173f52501f0f19f2a8afee16d6cf3e61794d44fca773 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0e79e977 | sha256-5ce8f6a14317c2f3e0354caed461b4531fd8db62995007134a16a322eec87c73 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0e79e977 | sha256-6a77c6648409e2d47d4aa6c87d68069bf8559560f770774722a6fe22417410de |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-4e4fd5fa | sha256-e9062190a5a764ce3d0143cffb002c31769e64da2c1c51d7cf7dd50ca7b453f5 |
