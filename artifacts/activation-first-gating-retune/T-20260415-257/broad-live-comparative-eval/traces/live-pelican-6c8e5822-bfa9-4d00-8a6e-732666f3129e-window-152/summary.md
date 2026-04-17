# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-152`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d734aa4f8ff91d619ea2bd69d87aefdd3f36d0cec38d3997b6f1c5ab56a102cd`
- fixture hash: `sha256-b5a8d59003130cacb6d12d20cb7f35591a0ecaec31de33844db54aba06f55180`
- score hash: `sha256-d7c643c0d17e61f48804132dff601731a0da1ab96d40cff244549190ddcf9f2c`
- bundle hash: `sha256-a4f49b05a0bf6859746177551d5ce77bd439e3077b21f7b9ef5aa80f0df5a64e`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2c822d5a812142cdf3ff00336272092554327fa9d0fe665c2253ac281723c371 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-12b1cdbe8e809af2567fadb585b375680f6766cffd9d923d3e2e548536d78916 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a9dea577f44d9b685d9d7402aa27ac99df5a248e08b94d3558e532e12ca8780f |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-67485ffa3905ab611e1ff01cda081bf67a99c2f1bdd85c416dc673e6da856cc8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a9363ce7 | sha256-dbb2c69e8d19be589a7bb1b1401728b629af0a3538ebeaa1cc29a294a095b0dc |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a9363ce7 | sha256-e7303c79b5f404fa9ba6889fe71a0f15ebb09767aab8119f10454faceabb59e2 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-b007f0ac | sha256-4434f34e87f87ed76bc0214f23844db41b26b054094ac2b0e4e812402d84ba47 |
