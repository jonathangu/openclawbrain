# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6ed82473cf8a44c6f378cb688937e33b3ea351a6801142726a4915ee5fb6d88a`
- fixture hash: `sha256-4909ff1896b085966400449aca0e9ac319b4cd9d22c11198c9e8e1d61fedcf2c`
- score hash: `sha256-5ac5fa8ac28035d9be58219698a7a0eb893798b479dab00ff266473cb1aa17fa`
- bundle hash: `sha256-73f04a45578019887f91fcb543cdd63f21902d890e5ff1f29c28995e45619cdc`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-46b89f268ea78eb5e49f9755003f2aa744b81e1b854e2ac1c9e8f1a95cc59955 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-25bc2bf0e9558a026891dbe4e3efc7b9b488ba1849d0b287d78b38a12631b364 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1ceb6b9db92cefa297a908c543c9063504f3613acca5cb86e8c87abf5b9fc3fd |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-248e9f80b3286a63db129a16fc41eecf9d6ca5b6e266ff414aa446ca9ee9600c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-4b2384af | sha256-bb07e38ffc95672926fbd7b754ab06c01ddabdd6f38d02db17546f67b553a84d |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-4b2384af | sha256-9fc48740c30a11236dc9c1b96030ac128eb9a0fba8c1e62de136a97a6c6b8565 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-9e4e40ae | sha256-971913c8d38a6a0ff9bb5a526cc538b858d2674581d03d1b78c7dfe5ea14d5e8 |
