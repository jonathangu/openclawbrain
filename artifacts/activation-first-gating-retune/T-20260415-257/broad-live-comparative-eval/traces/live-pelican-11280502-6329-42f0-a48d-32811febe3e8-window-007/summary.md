# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-dcacef92b60f17d22d3d52374e44f238a440853875f12c7d1e583783b59ae36c`
- fixture hash: `sha256-0309e3e5061db0b2681b554a0a0a11c1546e71f80786bcf376c32c5e8bf3ada4`
- score hash: `sha256-fe7f8b01f8fd9ccf4f7979741c846f1be94a6f0331d5207771c5694a0468f5d2`
- bundle hash: `sha256-eef430b4f67883f880f3c86b32533a31cb1219f3b60fe7cec7e5ac25de8bbf54`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-23b94bf0f175f15338e42b4d068d31347ced04aa9e1b9081298f39e373d2dc34 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-e2ffa7578325096e969f7b42acc1696695a74f0a0bde11f299af51977f04831e |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-a05adea9c042d5a9bf2fa133b65f869c476c2079b931f318b225d8d5bb34b1b2 |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-cf869076bc02c5fccf6234a4dcdc86b4d3c127541ffe03d5392809f9c5d8a359 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-d5ff1665 | sha256-209f47cf4f897dd46ba9075be6337f84e8023be28ba97f68cf0808aa7a57c5d7 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-d5ff1665 | sha256-0faff769bcec703213085f6309d60d0886dc011e22d26a933b461273673c11ac |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-f05a677a | sha256-f625760872e93bfeb736057bf6a497f745db7521bec939ac9ff408a6a200c37d |
