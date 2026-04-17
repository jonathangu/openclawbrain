# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-001`
- winner mode: `graph_prior_only`
- trace hash: `sha256-02a1da575e3574b7abbdd906c3a2e763180dabd7aec44faf80aa583f38ef8508`
- fixture hash: `sha256-5054e6a7c0e886819d5cfe411a8be2314f2663654e1f3235054d1d832296503a`
- score hash: `sha256-8671ff45faf52602a2f2e5ad6a4d91460a203a8c831e31221c2a8e62ba83d20f`
- bundle hash: `sha256-4e8952aba4cc596fafe820a4e44cf10dd97153d7293be86dbe18cf5192ba17e9`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f0f3e8434e68958d057893f328bab1455284fa045d15742e91d115a9a3a34202 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7232195dff3ae3110de0d2e726bcddbbc86f110396e64ed731734acacc179226 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2f58b429b8deb87cf859600dc9384a6161b471410291fdfbb325c25b61149a7b |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-4eadc9e299585f78bf2cae03fd5961dab68c5de3c323b085c0ffa8a428908348 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-cf84ac1f | sha256-1a332a341242fe95aac2b3567920dbc074f232d805f2528dfc1c429a92112484 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-cf84ac1f | sha256-1a332a341242fe95aac2b3567920dbc074f232d805f2528dfc1c429a92112484 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-329424d6 | sha256-9903784db06572a8e9bc91a51c617012ddf95550edd458f7e0d4d244afa2d027 |
