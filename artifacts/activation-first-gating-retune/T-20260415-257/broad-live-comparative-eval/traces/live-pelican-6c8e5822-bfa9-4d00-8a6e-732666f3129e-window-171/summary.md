# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-171`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f2aad77541ac9575f5e5ca17b331150d26a5ffdab9f43024542cda1cc603e5be`
- fixture hash: `sha256-bd1f8b0e0683d35bf0b6cddabbcb17bfbeff749dd6d56a3da4fa75988fc68560`
- score hash: `sha256-dd57b8427ddc9b27b16a8fd58f4a1b411ebc057b48486c724a0b4188585622f3`
- bundle hash: `sha256-724eec5618b03907dcf7e3c08839e288676e91d940e2cf3ce564bfd8c2cdcb60`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8b6dcd51a56bbf9edfb3ea54756a6521b5761e2fe2a8b04b095719a90cd986e9 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e9e915218cec6a9bc384777d1718b6ee439ed7feb012426633ba9d6fcea3d63b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6aaaedac8b2431a4f85fbbaefdc8514328fb9455f26b65560ef0e410515d56d6 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-22d0d52683b4ea583b825f751f43da3498e2e14fd544e034f8a1b88aca5a259e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-119910fc | sha256-14fd6821fa496f5782c8d746b2f99f057262122b15d5727077001b60e7948a23 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-119910fc | sha256-dded661e4b7db006bf180df2f6b592962a988ff1add05f26fa14b3c1c86dc767 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-b8e250d9 | sha256-9c08825fbef107496052a7f4b34583da457ab354a235f21f242f5c9181a18e81 |
