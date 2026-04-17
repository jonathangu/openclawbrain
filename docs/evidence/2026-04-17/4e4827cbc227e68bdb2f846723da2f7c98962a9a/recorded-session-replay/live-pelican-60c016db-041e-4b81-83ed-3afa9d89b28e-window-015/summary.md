# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-015`
- winner mode: `graph_prior_only`
- trace hash: `sha256-19cd6a701f3afe5404567d59955346d7cfc26c77deb7b29e61fccacc22d3bbfa`
- fixture hash: `sha256-4dda7357e5652f879faf39fc4f606d23e6674326c96ea6b533ba27ecfc72cf16`
- score hash: `sha256-525a36be0b2239c8e11edcfbfd67d4f81d79fcf79620c5f403c3da626eafd80c`
- bundle hash: `sha256-39cc6f181db2049bb6550c8efcc019b53a54621a5053a32dd3c9553330a80c00`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-934729bde748377658ef5251e3c9784137a24d5cc133cff448c2ec475fa6a4b7 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9707346cdf340fbec79b0f9a30599230683f9d333e948b847bfdf016224decc9 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f3332d0ce53fcbd65f1c7ec6f99f20746e1a248f8d4efecd4a14c88aa1a14cff |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-2e360a073cc054251bf37511d9a422ce6c5c03bb1b2f07edd1d665b0084c348f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-143ec5cb | sha256-e0edfa59f645dc2630c5ca31f7435af41f2f686c1511d98f3a7fe6629e61b4a0 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-143ec5cb | sha256-8a06da1e564d10b953a6f8c1ecc049e29ec034e37e6321b6c0a33f387e46f733 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-f7ba232a | sha256-cf80772c654e1a1a0fff8aa67bc99b89e00734e935c8dd4bd44fb88eccf87565 |
