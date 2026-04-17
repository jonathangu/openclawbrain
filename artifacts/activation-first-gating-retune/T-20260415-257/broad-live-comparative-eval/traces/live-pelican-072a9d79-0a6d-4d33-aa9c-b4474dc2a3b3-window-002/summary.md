# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-072a9d79-0a6d-4d33-aa9c-b4474dc2a3b3-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1df02420998a45ff18b6fa7592e1d6cd553e69e00670f629819f48f156232f3b`
- fixture hash: `sha256-d26e41f8ffbf777f72220318ad80ec7f532c81cc4e8c86beb0f89befd769d272`
- score hash: `sha256-aef5b04a789328626bf661d8942c5aaa07d849a8052f1e9f81f8e5d05e8e3661`
- bundle hash: `sha256-c82e9a237a41897ce1216907cc5f2b706e903b126184a45dbbd56b9bf5016c43`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | vector_only | 100 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 6/12
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 1 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-caffae64068969fa7e1d950417642498125ccd7a52b99fe5538a0a0e555ac8a8 |
| vector_only | 1 | 1 | 3/3 | 0 | 0 | 1 | 0 | 1 | sha256-de550e40ae53834f90f9c665c32fdd48331712c74803b2b45698c579f105f642 |
| graph_prior_only | 1 | 1 | 3/3 | 0 | 0 | 1 | 0 | 1 | sha256-18bbcbd7dc2e16b2e8072b7ba8913a3a444ae57cb660907f78e4503d61d43b8d |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-24a63ba979cf23a438542454fa4a621cb7abf153e9c1ceb01de16ec9878f2fa3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 3/3 | no | no | pack-90d49702 | sha256-2a5ef103b4139cd4bb58b85bef0a7530931c30df51d306dede889da3954d32b1 |
| graph_prior_only | turn-1 | 100 | yes | 3/3 | no | no | pack-90d49702 | sha256-f7e8269d1ce2bd06c13a3799c1e0d126d94b5b858daae6f62cc0f9dcb514b799 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-d08fb30d | sha256-92d1b467b8d156826ef69d412112aee5e41c818d986c26a6be78a1f44b491b79 |
