# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-034`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b1cc11a021b01ca8aba99240d6af1922e93eb57eb02a859f34b6e95e9d51abb2`
- fixture hash: `sha256-25597cd73d9bd0f0d086440e05e70594b904a1395b870c35b683a0a720d202a1`
- score hash: `sha256-143888518c0107262cc474d35dba24f9c674ec79318ef6a603e3c39986aa7c8f`
- bundle hash: `sha256-30d113879d2d29f70f7f13cd254f53896d2fa5c2ce48a4ff96641bf23cb9988a`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-8e0e779f7703426d9dc6e56462e7c187c6dea02bfd7bb266c409658015b58695 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-a401894e6066fc92b57099218bcbd1d6817d7f5649c9341917112b7e6fec2fa3 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-adb996a66f7e5ddf8be00672d4042bd44f1f60c6f02697ac55d3553c874cdfe7 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-620ef0467e7512a8c1b64c8b798d6bbc21810e8d4521392a636a34aa318ec29d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-6268c3c2 | sha256-55459158e658c7e9ea15253d3a5059c991e061ea8238da74d313ca8179f80d35 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-6268c3c2 | sha256-abcd1b1244234c5de819ff2402c8be5484a57c67212b5a2798f838132a92a3b4 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-6268c3c2 | sha256-55459158e658c7e9ea15253d3a5059c991e061ea8238da74d313ca8179f80d35 |
