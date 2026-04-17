# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-013`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ad8a200767aafe991ee7054e677a19f37758804d5a9a487f59ccad4263c83187`
- fixture hash: `sha256-23ce3445f512fae9ac35202b97a34c12c8d0db3c79197541a8b90358597638a3`
- score hash: `sha256-0450a8854f4ea885733a8d84182857a8409231dac243c9b0cb4f6e6caa848d82`
- bundle hash: `sha256-48c30e71cdf57c7264b40aa002300ada60d8dc7191044d597268e56326c8ad49`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-3b757ba8fadc84f09e0e7aed31f0b4ebd54fa8fe354fc559aafe046aa0541083 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-7bffc458f0b91f49037ed5d462d660545161d3dec5395403ec7771c2c63e2890 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-dafda2aa80bf80ccbb99af669b89795cf4a2a233809c74c731bc799a19bd579d |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-6c7cd08aa671794273acfdb15b2fdb53f207523e9adae3ee2274c222cfd0680b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-ce7e923e | sha256-627821a9911abcb32389321b131795847f8a1b66d8aca649c28f85fff5107475 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-ce7e923e | sha256-d48bf8decf22db406b58dcf66f0ea9702b5c16283ba091bbfdb4c7b9d0816604 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-594388db | sha256-8a7ec7d4e01d72d65054502d3c12957051b58269f579f8dfd1cecde441a59f5b |
