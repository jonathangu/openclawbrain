# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-011`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b2430ed58ee0abca0aa0224af405db6344da7702ccc6e754dab5dc0867b7727d`
- fixture hash: `sha256-0827a1eef5713f16e574a6c5a2c4721f6c9b9ebfe2794b2f08af42e8c07ece50`
- score hash: `sha256-3109861a179f8b379d534b10d4d42a3d3d6cdcbc180ad48f7c507f4e6bebc1ae`
- bundle hash: `sha256-37e115ee7d1a6c3d4860cf84421b6a62a855d85e59cd67245b8d7d10bd7da5b0`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5d95dae3d2cb2e3da5df09b63b5296f231dee9a351a91285d6a68ab316bef562 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-099df41ef53f08d06b132a56ae0697e8d3d197628358112f482e1c7a8344547f |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-15ee279678bf784d14bbedcec19b7e361bae21efd25bc6505a51d2b70f2b8ec6 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-d92ad8cb3fdb444f8cb33c663867f30dc6cb09523dc6ba665875a097322d1b8c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-246dddf7 | sha256-07da21f8315e2ba66041e9591b95ce5f3d7a1c496d81395ebac8baef9e6242ea |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-246dddf7 | sha256-079817460ac741e66268d698c3cbc8dfbbc85ac41170444e954bf498ed2b6f50 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-ee5e50fa | sha256-1baef5b24cec228007a0e6e29376bca2a5411696ec36f1bb6d8f5c471731cada |
