# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-029`
- winner mode: `graph_prior_only`
- trace hash: `sha256-24701235d9bef68e6850974201809e3a73463fe7ddfd0b5cfe74a867885dc71e`
- fixture hash: `sha256-7c9db0ae094c3de40db6d4e0f20c52b15a3dee97c3144a7a4c433e3dd89b20b6`
- score hash: `sha256-366d8f61b462b52ae8f571fe50161139262b3fe0c2951f4540e52aac0e7f3fa0`
- bundle hash: `sha256-2e21d8aa6a57c7d2a8ab1c4a6e98b4590ef447a4c584fdb4dc5b30d3867f8bb7`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-0f3c8d6c272d7556d73fb57fae65bea8046db993f5ac8290705eae6ece09a508 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-092151d0332decbef653a98d57d8a19864c49018d68fc5644e3e2f37c277c517 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f83e5ef7956f74fe3aab2f2fe604ffd70321b2ab39c198c4169ce738f9558aba |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-8b3c9d0123e9fda2d8802a091bd5cb780fc5509ec816ddc85790eba64c2d4745 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-a6b469bb | sha256-75489b4ce3fce719d782d536162f9ead4f4472df88525a47ef6c1d15d80c940d |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-a6b469bb | sha256-f18a7964913e2411bdcf8196b209c546d9a259d1da720de79ef0f0e759f5100f |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-443b396e | sha256-44d5f1cb72d09b93101c4b390c63fd3d634bdcbb39ae952c56b004d214205392 |
