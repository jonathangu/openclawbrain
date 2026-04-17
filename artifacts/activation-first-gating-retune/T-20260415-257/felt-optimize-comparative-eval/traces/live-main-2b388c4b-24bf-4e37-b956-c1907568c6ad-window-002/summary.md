# Recorded Session Replay Proof Bundle

- trace id: `live-main-2b388c4b-24bf-4e37-b956-c1907568c6ad-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-a7dfee8569cbeebae47062b33f26bd559b44ab7e32ac7a65f3d53fcd4f9d6446`
- fixture hash: `sha256-bf2c49e43d0148934d94e443780f19f84be1befb9f46554500ee32090d69fd0f`
- score hash: `sha256-1f7214d8474dabf0d9e77d094879dfd904958d0f155419195e29fbd968c76ec8`
- bundle hash: `sha256-bc67671cb88662a20b84522c24a07894545f9053b200f3dfa654b3306de84b4a`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-aefc644f475f6e64faeecc10e1bad33424cc557b74533b3b9b16e76adc362925 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-5a5316a6fafa4add55d84a7e24f78eb3e520d84eb1e484ee82f061c47e8c6546 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-dcb516b2f220a227cba1fc3af269412fe89aa54b1d58e9ae4388d1d6be53884a |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-e947f6a1d9ee16e9ba57e28f8ee297359b2c3cc7b56baf97d443f5723e593903 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-2ae7586a | sha256-7b2341d5a00349b8c318a2bcab31545d37cc812f2105e0f037ac8716517df997 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-2ae7586a | sha256-11c671b4a6a1c3c8316f14d915b5f36f124c4ea9eb9b432fe14404065ce6d65f |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-7d9bb11b | sha256-780aab512e877f98cc8894b297b4cbba73b0724440f3cd90cb1ce1ddbdf735fe |
