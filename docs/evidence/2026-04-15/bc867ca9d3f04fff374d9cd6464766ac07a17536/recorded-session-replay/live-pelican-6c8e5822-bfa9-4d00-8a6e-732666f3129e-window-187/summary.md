# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-187`
- winner mode: `graph_prior_only`
- trace hash: `sha256-eae72c8906ce053ade6bf66b6f03ddc87f48a19f8e1b50fd6f47ba9774ecb440`
- fixture hash: `sha256-80de414b90b70f70f1d2f2daf70e3430dc27d1af7b593fd0e1e1dfcb61676ead`
- score hash: `sha256-2d1ba379a5bf2e335b55d98672c5366013fef2e18d3d7182f2b0e0f8d8ee8394`
- bundle hash: `sha256-3bea89f100879421c40c0820f9b7dfbcfb6091cc0fdb3797c5f2187724b3a3e2`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0e2afd5c5e27e893dea21e62c6e8b163bef7241aac8748bba68a4d993b31b8a4 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7cac96bfcf692cd6927f8faa581f48769a812b2cedf85d5e0894cc992402c998 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1b6bc5643145be9ed3b1e0bd3d907b30513a3a27300523d5314b992362a6a3ec |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-325b93faa70155e7db7bfa895239ada6b541aa7c45c680b168585f7824347431 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b1406ad9 | sha256-86c2fb7bd75a868006a4065a363985c21acf077ecf5a486a2569df3ec4967d87 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b1406ad9 | sha256-bc71ac4d3b50d0b61ea961a40216f21653f4543c859f0942a6fbc2513cdb4c0f |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-b1406ad9 | sha256-86c2fb7bd75a868006a4065a363985c21acf077ecf5a486a2569df3ec4967d87 |
