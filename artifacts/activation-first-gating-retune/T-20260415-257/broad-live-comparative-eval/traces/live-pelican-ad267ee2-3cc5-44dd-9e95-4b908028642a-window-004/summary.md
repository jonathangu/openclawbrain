# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1382603626a2aff7d92c871c45318305722032a646fc01502f912f8472d0ed38`
- fixture hash: `sha256-ee8d3f8c272648220db4d9e69e984cdcf85084bd085927ab6802512d77922517`
- score hash: `sha256-272648cd532a05962ae25e0aa9e3e98561865df6bfaea7fe86a48bb8be0fd677`
- bundle hash: `sha256-deac5883d1b4d3521488ddf15c257a9e3be2f4366e56205400fc29c2984c55a3`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-394d692f2aa5412e9da10dfc0baf182beb2043f517fb99b07451a27af9201624 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-69e2b0de7229a3b82293fff7db2fee057b4662ba37e84bb5cf528eba18e64d17 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e8f06e343dffce7c812763e89c12c331b2d1c08917dea4df7e3314d64798c4ec |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-41c0b963853d39ec9a1b49b8092f4f35d79fa21da208a5bd6c9af9d7b033b4f2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-6af4d121 | sha256-473b203c9928d0aba79bc1df5756bdcdca09c0c9dc7c84e7a122340513b148b7 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-6af4d121 | sha256-119425e804be32ba3c7ca83a50720e7455be0318d5a728fb7d8f5b36bb578e9b |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-601897ac | sha256-3ce29104c54690390e33d4873d543b0ce49db6959a36de8c8d7d616386e836e4 |
