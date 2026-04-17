# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-016`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0e0ac143317067e59f64740cdb9f819c48d2981153767f573c0e73b22b2b7c81`
- fixture hash: `sha256-dbbac8f5cf8c52842e2689d4f90634fa33bc0bae1bc0d3bfd9ad2ad85d720253`
- score hash: `sha256-83aab819e8ee95090ed16ee294b6e685ee33b9a9ec58ba47f54ffa3e2710dc9b`
- bundle hash: `sha256-e9644919a290a4fe43758a29eda13352abb416caa0da691b0b33d89885477f87`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/4
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 1 | 0 | 1 |
| learned_route | 1 | 1 | 1 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-745541acfd3bce8c03c831feeecff054c455963b939319f1092513f43c7bfc25 |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-def53d865b9a1d4739cd9e65640f1aee969aad86616b291f24b3366b39f14cd2 |
| graph_prior_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-0102106984039e9956f477ac34c919c7074aba4107d2e568e1c7d23f05605c08 |
| learned_route | 1 | 1 | 1/1 | 1 | 0 | 1 | 0 | 2 | sha256-22c4b12063dc20209c73d89c26cb4b9eeda47a017acbedb9ead1665cdc93b227 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-ab0e34d8 | sha256-da8e17710de5ee5d0bebf9cc92790d18f8240010f24bbc385acd90a652321689 |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | no | no | pack-ab0e34d8 | sha256-b897d46d9591db5b53a188c8bd7bcc9a6edcc8a124dc90b459ec9544e754f38c |
| learned_route | turn-1 | 100 | yes | 1/1 | yes | no | pack-cdc572fb | sha256-3053a58540fb3aed03893e3f1f0c371124a250d52a2795ef98d408ab18f0662f |
