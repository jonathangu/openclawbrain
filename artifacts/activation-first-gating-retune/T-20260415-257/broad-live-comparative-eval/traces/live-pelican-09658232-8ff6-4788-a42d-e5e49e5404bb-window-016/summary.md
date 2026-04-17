# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-016`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0e0ac143317067e59f64740cdb9f819c48d2981153767f573c0e73b22b2b7c81`
- fixture hash: `sha256-dbbac8f5cf8c52842e2689d4f90634fa33bc0bae1bc0d3bfd9ad2ad85d720253`
- score hash: `sha256-45dfab8dfa8a470af70af72aca0468419e5ad6fff821d8eb101444ee8988e5a2`
- bundle hash: `sha256-74a1d7145c8f2d01072a6c152b1291fc40134f2962ed0bc7761edf137d7bfc32`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | vector_only | 100 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/4
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 1 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-745541acfd3bce8c03c831feeecff054c455963b939319f1092513f43c7bfc25 |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-7008b808eb865da091f86820fed1531802696dab3b9db2e70270f6bae64d9254 |
| graph_prior_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-d571b67b37efa6837564efa46a6849f230f14393a2840839e61aa9ce1df4ede6 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-8ac4686f2f51eb835b291932c7f9767459d5ba74b655741827b8957d46a1eded |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-3c0dbf13 | sha256-53e53e671b90719c1e62ba73643a85e94078a04b7a460fec252d65e8807b0bff |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | no | no | pack-3c0dbf13 | sha256-4fb3d06636113e86927a320f83e1b5e20a8b629bfe0d705962f7172788700a40 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-5ec4fd36 | sha256-93e75db6632f85f96e72b29c02405b00922dfc50fe4c597df05f26f02b06da55 |
