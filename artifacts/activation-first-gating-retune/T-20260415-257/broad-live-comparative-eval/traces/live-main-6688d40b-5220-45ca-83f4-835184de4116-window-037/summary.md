# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-037`
- winner mode: `graph_prior_only`
- trace hash: `sha256-67dcb8532f54fc5f6268aaf2cd959dca249a5c5832d88990a647435a45026ce8`
- fixture hash: `sha256-53a4c9afb3f28aa87aa1b17aad9db78e9f58b7b80cd2cde3904a19a0bb713c36`
- score hash: `sha256-67bc5b08626f03bc0a920b134b495c08984d293beed3e36fc0e0ec58503ec42a`
- bundle hash: `sha256-16d27f02cb9436ee016f55e15000e46ea825c9eb365932756d2a3336d793b7fa`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e721a4b8ae2bb9ec3999c909e1329c35bf2b76bcb692645b1624780e9c7c3c31 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c0b32b436283a7b1951f697294f5276aa2bf8822e44889afbdf5547cccd0abe2 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9213abfc49389508d91159cd529693ec0e3280c860bf5f27265fd2e472a91465 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-f1bb12086c52156ff924b599eaadc450e793760df45754e41d686ed461be864d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-eafceb9e | sha256-44359593acd7127d2e5e873fa558b1608b5454a7529f676eda70fc285f824fb1 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-eafceb9e | sha256-dec94b74c6d5cdddd3e0ccafa229868e5d8d5406ea57794e3cc6243348fc0995 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-c66f6ca9 | sha256-272bbed5d32ea98b2d36986bd60f3c1ce5ab5e08e8ef0557d19d653c7cd627d6 |
