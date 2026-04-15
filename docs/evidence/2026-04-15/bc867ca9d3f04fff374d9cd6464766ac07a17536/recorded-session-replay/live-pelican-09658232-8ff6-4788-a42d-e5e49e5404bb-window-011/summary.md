# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-011`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b2430ed58ee0abca0aa0224af405db6344da7702ccc6e754dab5dc0867b7727d`
- fixture hash: `sha256-0827a1eef5713f16e574a6c5a2c4721f6c9b9ebfe2794b2f08af42e8c07ece50`
- score hash: `sha256-024ecf29cacbaad0241fa91d9e825edb02778c7b54589cfb7ac09a58ce6a8c2c`
- bundle hash: `sha256-9eeece0e71afedc253cc8dffc69bbad136ea2491be4cb367ba15cd1bf7f099a1`

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
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-aad3d6000545148019e072ccc0714269eb4bca85f0c7a66c8a18cd75aadc5045 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1a2f6a9fd71b2d79ed6aee62d4351a0b6493fc9b51a8465f662ef6f2a599d4c0 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-9aed4dbffd1f1acd2b379e63760588d9c19ca3444dcd704d74d646ba6912297e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-53346361 | sha256-4b943ace8959d68766550f4d4d98b1d06889f2728bc1246a38d60f4be7ff2408 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-53346361 | sha256-0a1e569d9d01aa98807dafa3f0435d88dbf5d7aea981da60903b6df8cf110f24 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-53346361 | sha256-4b943ace8959d68766550f4d4d98b1d06889f2728bc1246a38d60f4be7ff2408 |
