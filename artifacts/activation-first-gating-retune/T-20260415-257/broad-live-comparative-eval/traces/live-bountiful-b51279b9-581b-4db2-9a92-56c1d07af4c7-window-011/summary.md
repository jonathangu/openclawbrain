# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-011`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5b1ea0e5b592ac9f44b7a4352ee387bb879a764114dc9ca28f8777b2e759540d`
- fixture hash: `sha256-64d81bc3848c97d19d6184af82f48df39e39a81124f70e1ee97b5963809c5506`
- score hash: `sha256-af7d1c391880754a12db684bca1e4f943cb800282f8c51367ff4e8035aacb749`
- bundle hash: `sha256-ebd435c0786ef9254aaa3626cd51ffbee0d363a507ce7e54122050effe93094d`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-71a0f39a2a308837d02e7c312f0b041017358409f7a91268e26a3ecc203deac0 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-51d714918d42e8fdcd57c2c64952d48c1c7be12d29175f095b623ceac94b5815 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e36cd09b41206a12198dad43ce2a1df8584f05e5994b6d184aa127c689964a07 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-2ee97de085bb4c0c0a811926bf3da86c83d0b083d8003a2071d73d0fb1305bbe |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a2d61a0d | sha256-37d51af32cbc6e14380b9132a9b531196fbd3ca838fa7b75da8f74d61f830876 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a2d61a0d | sha256-048421db10ac44bfb30f5ee4175e069aa31ed3b5ebb07cee4a45a9e725f3647b |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-efd8fca8 | sha256-8ece610fa4e929c5c6f033ffbf87681ef0df3f101e882bb9722e3222d92e882b |
