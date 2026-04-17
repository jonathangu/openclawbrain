# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-186`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e334b58e5431d3b20f7572c904faed7d64f26bc6fd3cb1bf1d055e492134e8a8`
- fixture hash: `sha256-8e788213c51f0225abe30e2600382afc50022c57de7f08753d94aa61dd287dae`
- score hash: `sha256-d378c5b93f08ebc3f4e9b0d15870c23f7935ff85e9c6837c271251ad2420eba8`
- bundle hash: `sha256-fb8c410922e4376c47902ca5ec5000b2ea7215151b1c15d2a652b587607dc993`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5f4621a0c949a3fba62d418ef21dd1d6c65fb58e546b35333db0f8e5c2c8785a |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e636dd77cfff1fd63600eea30a51196f6c09de0b62aa5c880a0bf1f4e0f1bd7f |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-41def882312732eb8a367d98009ffb6541433ce4339d0ea217eb36c57deee8e2 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-be8b6fe00eb2af88f96c24777c20c5634c75268a67ced5ea8da47ae800c548de |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-4e0b504f | sha256-6bfdd9644f2c949fb56618f2fa2161fb6592b506911c9bbf3802c9c0232e4168 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-4e0b504f | sha256-0c45e1dcf8799b4034fe8388a2d8bed72a749da0ccfab05a47fa512124870c10 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-7964729c | sha256-b99e55a85bec11f527781627720325465ccb0161eee256f86f2e3c5a6b52de87 |
