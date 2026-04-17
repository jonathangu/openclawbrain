# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-017`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ed2f9f31e28bc4c542ba13fc0a4ccba3e6b6e5db3982235d09f16d62242d7c5e`
- fixture hash: `sha256-c571aef0c0ac7b60f97a81ecefc88f95d1024f6a761836a503482febdda1b1eb`
- score hash: `sha256-77ad94cdc284d149aa415046ff39685e1492e3127008cc5f82196f89023947f3`
- bundle hash: `sha256-07ccc860857746aa4d1506dc2ac141bc1d1cdf505c23826cc9a21d5dc6db4f1f`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/4
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 1 | 0 | 1 |
| learned_route | 1 | 1 | 1 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4a371a11e3f0400310e154f8ea3c13a532ee5c397c446eff3697fe01cbdc026c |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-334762c949c5cf6ba11c9fc4a431c2d77be98d401bce73525bae8cb66aa253d6 |
| graph_prior_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-2ba9e565eeedb48b008b27368629e69cb56e7472ac559d53684836af37713998 |
| learned_route | 1 | 1 | 1/1 | 1 | 0 | 1 | 0 | 2 | sha256-c3dc126a56f145a7c4faad1b762216506c4d875854de4e9eb6e6f3df6ba0ee89 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-12cff63b | sha256-9b3f08110bfbb63f1d66119ac9c64af5fa2afc11dd0f4ff9b4b0b36e70e8e7db |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | no | no | pack-12cff63b | sha256-25523faf1870ea3447c36ae02f2f72239e882b8d81c4149a6855d1efdeb63c91 |
| learned_route | turn-1 | 100 | yes | 1/1 | yes | no | pack-7c395efe | sha256-48d0a73132ade79e654af10af7b0b714b7d1014fafdf6c47009ade614cb27edf |
