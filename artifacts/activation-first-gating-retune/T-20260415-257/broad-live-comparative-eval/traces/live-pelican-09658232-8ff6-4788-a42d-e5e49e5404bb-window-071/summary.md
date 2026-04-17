# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-071`
- winner mode: `graph_prior_only`
- trace hash: `sha256-aa6afe07711fbc8a13484cd14e70ac82c78cc503ee5449452a36b775fa63c3d1`
- fixture hash: `sha256-bacff39860081979b6852dc7223e7e30d3e6e8700496899a8864e78cf3c36fa0`
- score hash: `sha256-6e5eb70280cb8ff7f65a2d3f68e229f2118659a184aa56614a1af376b4a5ff30`
- bundle hash: `sha256-21fc9bbc6fc2fa1c7c5d482d670ac3f0efd463e8dfa0014ac6dfbc3f4735a43e`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-080c0791d3c8d4b27935c18a06ca48413df84ee848ffe0bfd6099d007a81a298 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-cb53b8f485c51b97d90c229db6da7289ea8282ee071b92e7b3387ff6b423a8c1 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-76d7740fbd4196370b9fd41f23632f1cc2b891d7815baf6747928c869801c843 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-41459f71155710392643cbdd110a32fb8ad423444267da1e6ffa319cf29e002e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-4e46efc1 | sha256-4ea2abda1da7ebdcd0ab18a996a49ba53ddc960f6ed2a434097c8f93a8f91eda |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-4e46efc1 | sha256-16d9ef9570dc8597043d50317ad96ce30c489f1dc229bbc394b17f8dfc651904 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-7dd485de | sha256-1b68d9ea8f6edd89475afa3ec8af62cb2e0a88c5e14900c3261a1c6c39915477 |
