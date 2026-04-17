# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1a576ab7fc82836d62896c5506ba892a7997f6c29eafb6387885075368088d2b`
- fixture hash: `sha256-e830bab1e1b5c601ab706b387c4f671be86f28c4ff56747b0f78265a86556170`
- score hash: `sha256-4a4724800c04b4ebe50c5a5c0a5e10a13fc63ee8ee72debe750c769d44dd2367`
- bundle hash: `sha256-68069d26306d06c693e265148d668cd15a0212f5dea98abc24b5e4b2cdf16577`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-309a76d5f65b7ffefd710af5c6f62a81606516631b55e10f450624750cad9788 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d5528632771f95b7309239afd2ef87e07eab350939d718a02af6e27b46c0cc2e |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-3fd7a36600b012b702571e54f6abd406e7e4bb750a18e539cf3b38fdb3e3c53d |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-7f037b8e0baa50d53112bba150af7ca5de3a107aa2b19066725cb6f0e43007a0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-e8455bd1 | sha256-1db62c73cfb105d7a25d6d60244a9ba49a01e5f38970dc45378332ba7da6a402 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-e8455bd1 | sha256-75b42e2ddd501b1461383be364e5eac8fb2a3c42898cbeb555879cf844f5ed91 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-7506bf32 | sha256-51317ec372fd5d39482bf55530d3a8d4dca67475fed1994ba33266a906febe04 |
