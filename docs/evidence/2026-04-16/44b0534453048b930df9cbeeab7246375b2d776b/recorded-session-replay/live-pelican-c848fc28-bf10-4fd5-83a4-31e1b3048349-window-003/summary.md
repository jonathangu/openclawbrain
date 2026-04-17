# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-c848fc28-bf10-4fd5-83a4-31e1b3048349-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c65984dab810fcd56a73ba24f7e48a3de3329e9e72c9abc055205970cf393432`
- fixture hash: `sha256-6edadb4cb34df6bab57971cb77cafbb8b923e3e92f73e144950ce412708011f4`
- score hash: `sha256-fa528811dc702fc76861b613ae96ae4fa08a9627dc1ddcdc8b4fbccd1f72ef78`
- bundle hash: `sha256-c733e81810363136d58f9a0be8a40307ad144b97f531dd404cc6aec95a12cce0`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2b98041498153f3fab8845179ecda7c5ad292ef71a993f916db2031745eb7d0a |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-bbc459415497e1137ada2479b55d36e5ccab55c6ad9343f866503c3dc0758b41 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-45d1aed4dc7e9188eae6d99efb62e2059afc6b1d37a120c437398cf71f2ab102 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-3a18f88c24b788721e131acca9ab677741d460043ad7f511d960aabc30336038 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-3163b77b | sha256-1d16096a92ec03150a6b5e646fbb4093049523a9dabb232481428eaa042f8142 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-3163b77b | sha256-b4f190a238aeb0fc49788a72606c873175d4dec02f2744e69cdce51e177be59a |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-e124578e | sha256-ed26df4e4bbd6bb8654ca8affcceccdc130aed758c00842106c8eb03488d0aea |
