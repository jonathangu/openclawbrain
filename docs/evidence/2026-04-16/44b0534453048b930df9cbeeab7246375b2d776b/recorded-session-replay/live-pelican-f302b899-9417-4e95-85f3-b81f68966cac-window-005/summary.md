# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6aeb45f46257078a73e31a3ca01fc811e5a3a9b2828328d1595fb41ae1cb1b87`
- fixture hash: `sha256-b90901422fe4620c22145acdd76fedd90d08a07ca2636957ff33166af8db8c6b`
- score hash: `sha256-20b842c35bd06311d9148b0141df854cb149f93282a2a8545554a883615ac11e`
- bundle hash: `sha256-91ebe2fc1954b9685f0a1c360ff70ede434e56b69d35a85c973c882b4a08be55`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f9c00df70ff9e588e665c6961063a6f0105a883c9e9bd2b1d2f815eef1057f7d |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b62bdf36366634b10f4e37eb94eb0635fcf10e98f82dbc570dabf2c71cef5c34 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-2495312925ac5346c0d2e45d3dee7f8eb501a34fe59c62cfe712912d34a48476 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-3da2b8beec8f3613157da1a125457d329b3f2a8eeac76510cd6a358f18a91bd4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-c8bc1287 | sha256-037f2ab6b2f8a9eff5358a0e0f905336b30cade3f4bfa42c48d0618a1fead115 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-c8bc1287 | sha256-b33f1a17d8bfa697d6f0f6b40538f4888c2fc192b4a9b8a1affac5fad101d4f3 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-1ea53dd4 | sha256-912a2415e086709cee88e6736d09902bb260b4e618dace7731aa0882f01617c5 |
