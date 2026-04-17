# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-015`
- winner mode: `graph_prior_only`
- trace hash: `sha256-06c88cbd7b40857f6269dd03d5e04022f7a27c8c5e2a225bc79b2768cb90fdfd`
- fixture hash: `sha256-6d62bb5ab6456b9eec73e20f3d1a35ffc14e9452a4f4442f3b56ae134f63d27e`
- score hash: `sha256-c026a7e75ca84b66b0ef83f44e047082d23501f621a1684e8625807ae193e311`
- bundle hash: `sha256-ec2fce79d0ccd8e1a87772dea80665bdd90e1d34b082534b694c736a839b9d53`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4cd163af8984e87c72885a17249c9a84973c54f74e5363d963d16ae86c9b4e43 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5e0d2a85561adda15f395c1d0755e7f1ad864812d6210a8d75652bba9eac43a6 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-11612ee35f69e7d656c69bae00415180ec3ba6660364e8fc236c07df8d214086 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-83efbf2f15ef4f84643af74f8325bafe62da421cbef6562a9bd148e40a3a76e6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8e6243e6 | sha256-b874f03ee41166b9f14239927c1f0bca4ecac433e7b3cdd45cff3753470c7721 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8e6243e6 | sha256-3cd0099e0a79a0d2a27bf6d8768469f200538979498c72dfa394a0d6da6965f8 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-aad44985 | sha256-404776ac92cf3af0347fc1a5c9cbbfdaa3faa051ec875249e7f7410567d96c83 |
