# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-187`
- winner mode: `graph_prior_only`
- trace hash: `sha256-eae72c8906ce053ade6bf66b6f03ddc87f48a19f8e1b50fd6f47ba9774ecb440`
- fixture hash: `sha256-80de414b90b70f70f1d2f2daf70e3430dc27d1af7b593fd0e1e1dfcb61676ead`
- score hash: `sha256-7196fbb16647c673771e9820ade0ccdabe192b30c71ca6a8ab46bcc0f0c0ba10`
- bundle hash: `sha256-624816decd4572b646c61ef077391f9043e0ddc641060974a48cceb603f6b6b7`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0e2afd5c5e27e893dea21e62c6e8b163bef7241aac8748bba68a4d993b31b8a4 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7a54694e32bc7a3024f60497aee2ae2543d631d38e16f9652955fe9f9692f061 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d9ab8e0f50f1f452de2042dbe4b60c65b7f142ca4c9f6df7f11d0b057c0590ca |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-36c80b5193fa9976877312bb4a24d475dc384fd9c233cfea6005954cbfc00496 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-2890b063 | sha256-b152d5638efe1f2ac39e64e2827eba603aa9ab885ff0fab4e1c067ccff2dbe19 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-2890b063 | sha256-b0b89e255fbd4001d46ac6e74747532aaadbd8622d063992d13629407e338042 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-d1cef504 | sha256-4546b70056d24c11a918f8ee96a4949e98ef057d65013a5bbe4acf1d6b018ea8 |
