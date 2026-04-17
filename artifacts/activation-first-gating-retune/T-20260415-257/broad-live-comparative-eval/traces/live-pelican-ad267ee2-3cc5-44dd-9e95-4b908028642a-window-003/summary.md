# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-154227e12deeada99188001de1f98c7859b44b0240a0a63280198f0600727836`
- fixture hash: `sha256-141e98c67b76e6b544c136b2dc9ec311316dae947241f48af13f9b3f509e9c48`
- score hash: `sha256-4a82be1dbd180ede413c3021cd5fca2e6c5867401124219ddeb14b2ac6a6728e`
- bundle hash: `sha256-bbbf02fef7675f8bf5dbfe7d77f18199a9d6f44e5a370ab5dd6f3eddc436e1b7`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5fd20c45ec549a50541ad825ca2263c2905bab11bd8f991e3eba1789bd6eddad |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-57339d884ad4328c10d08640a9cc6aebb3fe5f6ad7b127999933b1298e7ad681 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7236ad1bbfd4d124f8c25c8dd460bb5510911ccb349a2c53c71518f4af71881f |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-8f9c37acde2dbf635fabdd8995f323db83b537d8ed0eb3f8e63408e32a23de5a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-cf42e2f2 | sha256-773920a8c433a01c62b426f62f9dfe81f70de06fca8287a2474468ee7157979f |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-cf42e2f2 | sha256-b70e453f78ef7fa0bd64e63b43bde04a5cd52fa0c885dc725a1913a5561072c7 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-3571f419 | sha256-5c5dd7e33e2bd0e21e65a5632284986eb4cb7f9ae2686e033d7d9f7d8759d160 |
