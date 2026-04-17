# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-016`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8d30d0b2ffefbdcd1e1a89d75d980761c51cd05c50f2c3cf1f693944186af036`
- fixture hash: `sha256-029c6b1d164f9bd1c4692f0184b6bb3b57e3ba2e59663e9c61a6962698d01e73`
- score hash: `sha256-f9ad179e2d20c7e7f04b620a0bd258044d60651e267f52d0c9d96339f827758a`
- bundle hash: `sha256-819ef3efb45d8cf04322b91e46ec3afd9cbdffa00e5bc2dd3a7062e9d355dbaa`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-6eb030a8259079868b419f4ae1a6c389dd22240eac5e867e187ea0fab1adf6c7 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-7f89f402327c349df7965817d12026c78d89235bcc0178893271f3e44278751a |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-3c3a74ef3b952dfbedf782d8c69e8a37ecc8dcb4af606c5945b75bfd1ed0c97a |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-3081637a96c511f7f89455c6cfefb89fe2d267bcf0c099088834f58fe140314e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-c503b973 | sha256-e89cbe617693d0ade944f923be59cff0477a3fef0f2f7be06a6b7c8e8a79be6b |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-c503b973 | sha256-d38a8f5d287a5c5d46934426d3ac45870d0f536db96b53ce11235c086eb66fb3 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-17eb64b0 | sha256-9e6098f48f4c76e16fc4195334915def92dbd66bd7a9164b0b41657993c224f5 |
