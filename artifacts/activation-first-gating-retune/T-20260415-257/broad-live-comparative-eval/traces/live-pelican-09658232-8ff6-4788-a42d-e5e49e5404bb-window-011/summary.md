# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-011`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b2430ed58ee0abca0aa0224af405db6344da7702ccc6e754dab5dc0867b7727d`
- fixture hash: `sha256-0827a1eef5713f16e574a6c5a2c4721f6c9b9ebfe2794b2f08af42e8c07ece50`
- score hash: `sha256-98c476060fe7316304a6c3381f3326221b8e59dd2b684e363f2f73ee04af3345`
- bundle hash: `sha256-c117a8ac2eea14d30a82e21286e23db5549eed169a454addc5164a0b433f0dc4`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5d95dae3d2cb2e3da5df09b63b5296f231dee9a351a91285d6a68ab316bef562 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-746086cb9b6808a346af9d41bd661211e48ec77c94c163b2f712477bfae1ac45 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4f5dc73bd85cd38071ae1fcb12d6e346ebfcb6dd535b939936ab45c91ece4f6f |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-d5a1aa95fa29e318b87ce30158bf67ac6c38265ab5e49462fcf53c5001b5d109 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-facc3ab6 | sha256-f92e8df9f193b0a194c9818022ba60b79c76d86c551b17340373b563755a56a9 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-facc3ab6 | sha256-3bb37fa5e3a598863809a06ca16b6079e4562c3c630f53d3376537b12a3807e1 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-c4bcadb9 | sha256-acdc6bc04c889070321548eafaffd4a3a930aae9c8592142dc4a103b1fbd008c |
