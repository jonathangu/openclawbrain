# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-021`
- winner mode: `graph_prior_only`
- trace hash: `sha256-13cb1bca8722ed39c54b48c9d170af84a0229da5a1be3326ad569cdcb6c86e93`
- fixture hash: `sha256-3649ce5ca20580b372f2a2005a8164ef24eb19856bac1831bacfdfc2aeeebd5b`
- score hash: `sha256-1025f6b5a2213365fddb8fc70763c54cb6504d809c5bd288a7d825961910fc3f`
- bundle hash: `sha256-d32551f5a503457dd43946044481a56715d9dffbba0f889d11d221bba2ee2068`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3fcc253b9510f29399fe22001359326c4d47b1fc87658fed51c53d2aa08bb9eb |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0f5de2e114b95f1ed43aca38cc6af7335d90e076987b648d40ef6e4ee1106d86 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f6ad721f95a0328a41b1b7bba50562634159034e720d9bf13da4a0ddeaf01d44 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-72c92165fe81b28f89340438c55d74b75f0f3f2ce7f343ef7fb8a27e5a8aa05d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-15d0e84f | sha256-6f5e814ca60dbad15cba4d90eb3d687017b47dd48fd3e21dd5f7d301747ee0f2 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-15d0e84f | sha256-343b59f78edb133ee96b812d055faf7fde1b470cd64efd78a1b4c7fea7c748b0 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-15d0e84f | sha256-6f5e814ca60dbad15cba4d90eb3d687017b47dd48fd3e21dd5f7d301747ee0f2 |
