# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-029`
- winner mode: `graph_prior_only`
- trace hash: `sha256-24701235d9bef68e6850974201809e3a73463fe7ddfd0b5cfe74a867885dc71e`
- fixture hash: `sha256-7c9db0ae094c3de40db6d4e0f20c52b15a3dee97c3144a7a4c433e3dd89b20b6`
- score hash: `sha256-aba824ea26a144e9d7b3a84e894aed7f9ac049c0425d471f715d4bde7ed9cb0e`
- bundle hash: `sha256-9bf7555aeacb3484840ee64008a530d4cc91fdf98668b16ed8b9e26d96e3e219`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-0f3c8d6c272d7556d73fb57fae65bea8046db993f5ac8290705eae6ece09a508 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-bb1a21982d7bab5c9de6a4d8718e45545b899d7b11ec55edd3c6fd42081b5be5 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-2124e6bd57f7eafe7fcd91fa2669bba341f696ba237ec0b04eca89fe13d66669 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-ccc7f3067af0a7e9bb88926e6c1f91690b50919c47ee222e5ad22536a2f9c566 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-55b8fc6e | sha256-56919b14ff988648b1d3b54512b3cf81a05dfc9cf3c0853af3ae506e32a8e0c9 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-55b8fc6e | sha256-ec41fe12c2aea02f551cfc1b2b6d5c293904131bb809123c0d32e14ecaf7944f |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-55b8fc6e | sha256-56919b14ff988648b1d3b54512b3cf81a05dfc9cf3c0853af3ae506e32a8e0c9 |
