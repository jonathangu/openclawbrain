# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-257`
- winner mode: `graph_prior_only`
- trace hash: `sha256-cc22db3aaa15315761f798aaec1df1acf278bfe86338b981b1d314f80e60f459`
- fixture hash: `sha256-6250124575745297903131838786e09bce6bd0b2285afd782515714f7d74a408`
- score hash: `sha256-8eee90f3686dfaa21af8cddc8b2c2b40d1152b40242f8ed8f674b2052599f477`
- bundle hash: `sha256-9c454b8b8ba631d8d55e60def65ee9ed50bb7454555c9333fcd92c59d617005a`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 70 |
| 2 | vector_only | 70 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/8
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.5 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-363a0ce15ddf219a167f700ba2217552de3446a99d080128478494cb795b929d |
| vector_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-adecbfdb8fcde24e49a1dc0b4ddefa8de20e9ec56f52f50bf393efa61eea0c1c |
| graph_prior_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-c6034eb6ad9a701cca4f285c5a8b395c00e0bc87f2faddd78b04ea80c9e718e2 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-62b83aacd574480ba19d0ca996701e283c35ea60ba6dc243401d47b52a9b12ab |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | no | no | pack-8bb03422 | sha256-6d752af141133753fc92cf302f2892bf7294a16dc10043536942e74b3a42d2ac |
| graph_prior_only | turn-1 | 70 | yes | 1/2 | no | no | pack-8bb03422 | sha256-e6e49f4e2dd1a514ad8e8916e86e71d34d93b0a43a2794c3082f077d652ad0b7 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-054a68e1 | sha256-1b6f133b7d9f883ca188bac2a30e1e770a1a1ac61f2034b5749c8cd1c3409a96 |
