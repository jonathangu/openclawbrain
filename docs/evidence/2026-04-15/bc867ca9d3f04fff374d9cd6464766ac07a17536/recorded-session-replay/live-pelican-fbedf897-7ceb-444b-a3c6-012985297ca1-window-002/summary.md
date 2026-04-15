# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-fbedf897-7ceb-444b-a3c6-012985297ca1-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4e6bcca0889e112786ccd30d6dc08d693afeab2955ee3e21db9f09dfe3094e0f`
- fixture hash: `sha256-14ff100a8ccae36fc1c57494dcba2b6e1338cfd708e5c890121212b4f7b539d1`
- score hash: `sha256-33744ea61f50c27aa6ff6deb57858e25d3911631e2b9f8122381808f37962f23`
- bundle hash: `sha256-969df328415f3f0e1943153b7f2d9e06757df78c6ec2413a046b42eeca865dfe`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-86a96bb1f89c4f625498603269cc86fe2157c50e9372e11582b94c39873a6510 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e207bd4c1aaf0d42b9aa2886b7255f57b73a0d32206f547e1e066e4b28ced3e3 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-12412f872f255f82f45d3e2aa4ac41dd27afcb004bd854f40bd7b871225065b9 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-7a999c54bb674c4c9e99ae73c4d24638e6b4205d642549b5a0a5996e370e10d2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-13c10bff | sha256-3e90b8f6cbe9e03a7d4faf6be8e13d2fe45cd4551d8bf4cd6e8940e396c7b2a5 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-13c10bff | sha256-0488a608c55a76696790c83e5b7d8a65742b7ee1c5d52fc240db8a45668dabfa |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-13c10bff | sha256-3e90b8f6cbe9e03a7d4faf6be8e13d2fe45cd4551d8bf4cd6e8940e396c7b2a5 |
