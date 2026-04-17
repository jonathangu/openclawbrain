# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-001`
- winner mode: `graph_prior_only`
- trace hash: `sha256-02a1da575e3574b7abbdd906c3a2e763180dabd7aec44faf80aa583f38ef8508`
- fixture hash: `sha256-5054e6a7c0e886819d5cfe411a8be2314f2663654e1f3235054d1d832296503a`
- score hash: `sha256-ec4e4ad336b6dc9c2592b03eb12ab62cb9d00ec3a566b46c550d806e9d2240b6`
- bundle hash: `sha256-cb8ef234df268e72277413c214d2c613a6b9ba7282cdf0491f30d855ca67307d`

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
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5939185cefa07efd4237c441d23ddee7eb504f5936438acdecb6cd48d15a4309 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e0e8e251678165fb8c7efbd75f30a8a643c51a9f1dff64a93a38b8e713c069fe |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-18bd5ecc4a43734a55cd0d9ab84d4e9661a4a76f07ccf4367a340c4d229ab3bc |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f826c03c | sha256-0b8f2bc95a281ea43ee0987d8dda7dcaa52f513447c88337e495f74406032eef |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f826c03c | sha256-0b8f2bc95a281ea43ee0987d8dda7dcaa52f513447c88337e495f74406032eef |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-5b3638f3 | sha256-b58432f46bdd2eb48e0b78025a912189bf0c253ff78d2faeaeec063d6f80fa9a |
