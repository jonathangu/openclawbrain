# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3d264077d440eb71a4733b77855076b6c8ed4b150a58a0fe30d7cf9c384f3d83`
- fixture hash: `sha256-f8dd49edf14b538ae37136d1260c31d4ab4f9bbd2ad10ec02ad026ec49c5e356`
- score hash: `sha256-06ed82bc4982d1ad8ae333a94d838c26dcb1a5dccfb961facee152071c257ed2`
- bundle hash: `sha256-0f9957b409e14ee62c3df384ae80e1af39a3cdf4321e82764a8426ae2d5dae02`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0ec5243a79ec6792abfa1dadf6e65f10d6e160b483bb5f843046d875ea86d177 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-22ef63f34b1b350a6e19800b9dfc2b2cf3fa20d9ae2a303d77cee99a0eeb19cb |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bdbf4a443e4c69bad7b3b9f88cc235d24106965724ed7ed371758a7910dc44d5 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-24446160e97c5b8f4b3001c444971b628be738a4d4a01e0441de5c4cab9a500b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e409efb3 | sha256-70d36a02874e8475b61b1e8d8f84699d1778788145a1a529693f74fa5bade997 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e409efb3 | sha256-80852cd6a42230a458b2beb903313e1ca03562c8f65bdaed45c87b4463afa519 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-c683a6c6 | sha256-63a04ed1f2f4b86f45f1a841b4cfb461b219bed0ad8935016c59f255c3ddebdd |
