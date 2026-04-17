# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-089`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7a2236f637704fe149867dcf144d671dc7a13fa94e04f98252bb7a94efde6a70`
- fixture hash: `sha256-3dd95fcccf0fb105acb53dbd74c41b44d30300251f8ca1b0c6b6f7ee328de982`
- score hash: `sha256-88ad2465fc4fd52947f185632742709e188417e3e285342564824c915248ef7b`
- bundle hash: `sha256-59b3023b5223219073f22e218794b9014aa1dab56f22321e368f7a52a5e124ff`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ef52e9f6d08b86a0755671620744d8fa71177a56d88b43c65d023da00ed4b3db |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8bf02b850f28545355606b1a5afb51a9a661e5857f067c40860fd6a8811fd9ef |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d80a888fbeb0c4f76f2f487c7f1334c749d499dfe2bd1d0f53a86ac81e263da5 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-c085dc72a74944eb66ffa2b8c0a1813e870aaced55cc161425a2a46a39d245a4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-bc4bb4f0 | sha256-35bbef275bd319687e01d330540718345b33930b29611afd725ae10222ef6496 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-bc4bb4f0 | sha256-121ea72cd126144210d9aea739a643c99163fdf46329ece5e23970f9f06845a3 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-5983b25b | sha256-44c38cced40098e68d73b4cad604c82692872461f785138c05e43a1b8e311c66 |
