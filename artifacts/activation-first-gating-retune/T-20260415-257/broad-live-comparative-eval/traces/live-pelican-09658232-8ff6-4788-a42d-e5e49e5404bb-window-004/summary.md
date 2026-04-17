# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-fd4a73ef0679d3bd5e8a41ecf8528eaf1056f459a2933d6bce7a274e1da6704d`
- fixture hash: `sha256-cdbe046df5ba47eb867d34f32f856111ce7f2bac423e41168b29efa3bc680b6e`
- score hash: `sha256-58f4ab05f4e1327ef11ff06cbe2172f521415f7b9ace2aa366a251ffb96c6464`
- bundle hash: `sha256-8590df941ebb660788424a907c78f84e68ee7061b996551eb5d093b6b254dab5`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-195c8562b43d566f299d3b4d568af19c059fadcd5ad0dc52c1779f850a2eeca5 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2cbd438edd29953a4982fc464f439f6f73f603231ca968e53c226ec6f3c9cb3e |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-cba93c29cd25f5031aa1cd413f8a9d52c0c833468be1a1cf28467413fd01cf4c |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-f50d2d289ecbb8ca7847540fc7c93f2a8df8c22faedf2b461ecc16e40487e1c2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-7a5fb9ba | sha256-d1c0fc01f0e1e172522fb72d874d9a415b0ba524db86036762c23b3c3446614b |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-7a5fb9ba | sha256-d344405931fb3383b7a697b7ae92b0bc7d7ef505bb4aab62c2f8147d01981024 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-b6f17f8d | sha256-1a0c3702ecddc94f25ca56388e8cda009adc06b3f854040fbf484ef04ca5dbb9 |
