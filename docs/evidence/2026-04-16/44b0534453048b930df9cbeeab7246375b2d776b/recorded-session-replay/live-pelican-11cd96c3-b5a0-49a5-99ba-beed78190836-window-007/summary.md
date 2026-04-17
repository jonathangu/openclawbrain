# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d958774a8fc5556f6b626cb2afd5141be38b390f01c3f1c481f5689e5c67765c`
- fixture hash: `sha256-bf711d8c588faf57d4df6088b8652fb030ca7a163bb118e31c3e2f2768cad0f2`
- score hash: `sha256-5fa8bcf7d036f710949155006dadc07357790a7d5e810a9b3dbdb505630fb605`
- bundle hash: `sha256-6418d2b38c9602a0d3ce867eeee168c365f1b61d160dcabc54c514f29b2b63db`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7f7ca30e9c8433554610f300b068b172fcd1c7c716d277545f4d5940081fb358 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e2fb810e63cdc4ca6de42ac32d8222eac52ee373d2f39a09da2a0e1001a601f9 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6bcb8b747d2b6c4e8079b5b0f902fefc96018ed55fabaa14c66334707e9289a5 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-65049019cc5ee661790dee8842cdc29ce80cc37067aee9131b7b71af2ee302f3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2eb68051 | sha256-e57ca914046215f114dfddf4f72a107711be432e45dbbd4ea8dec7cf85f4af6f |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2eb68051 | sha256-665081c46ba57feba81046063cc875ac6274bc438091bbd67c0457545826108f |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-d91415b0 | sha256-eda5b85b66adea5ce1128d37ffa257168b5b586e708f6981007285a04f2eff30 |
