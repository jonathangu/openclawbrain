# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-024`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b224dc602b7429463a9b2fd5346afa6d3382bb3fd84bc9d3cceb0d3ff24896dc`
- fixture hash: `sha256-493fd471e0bb608979cd024ca51b9104b86ec7063e95845a4d6e7076002d21f4`
- score hash: `sha256-03d34778f56d260596c6019c1246b71d4299aae45dc6e2f1484ee4c876115ffc`
- bundle hash: `sha256-01f62d51b95949571aa4ff283a98ca3184b69d1a5624c92d7aaa5fa69a777f6f`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ff276f984ca7449fbf40ed52f8c73e2aedf05be900e45cdc0a8a0b8a46668591 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f60a460bdf827ba20d7a8ce7f35a40bd9a1410ae9266e0c19ccd53479aa1d610 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f85ee4b9150baa5da55380ab50e2c78c4952ccd306c3e04730d2634d68ba519a |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-04564111196f8e04a8cd5ce09652e6f3d139cf80978f178cd2502a6c6796d267 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4a7a1e8d | sha256-df60fbf135a00e235494340af12287a1084685c71517b64548c8d6bb524c00ba |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4a7a1e8d | sha256-17873447dbb161c845e01573d51ab23740200f8ff346f34b1e336bb613c8bd99 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-5b92ad32 | sha256-f9c1ca825b310213bdcd2e6bb108799775f24c80ff89d6ac58a77289cc14b087 |
