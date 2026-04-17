# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-188`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2286f1962f858995a9d11d68ccd4ff744be8c0925ede8b9595870bdf0f8216d1`
- fixture hash: `sha256-7131fce3dd7f89b87927812976c9719dadea253e34115e7f37e0887827e9427e`
- score hash: `sha256-7f3f97006c95ccb24f73cde08daf8da80097a05441e83afd0220a02e27b9f090`
- bundle hash: `sha256-47106b418c9a6bb29919093133df450c57a9e2f3881181ad3b8d486d81df9ab8`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4cce362e58248df18caee133abbb86ea37c7c8cc312d9027b572d5a719da7a87 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0e30fb04b02059a2e3138e2a66a4239b671fa69e9d40427f65c7584782253203 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3eaaaefeb515d77675804ae06fad2abeda64dfcdaeb5ebb62472f7bed468290e |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-f42d3dc381f5fcf49e97d9287363bab1a6a38387a01578e4de39c0ef4099c873 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0893270e | sha256-0c8ca261b61327eeaac0c763cbdda33a0bbac2b32d71c700140fc7b539d21824 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0893270e | sha256-41b25affc45b015d8a6df9ab2db4afe48faea9d604f2108936e52ea173e4317a |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-dbef9309 | sha256-bbc99357fba6e01b927b2e3c4636331d9ecdb986746681073babf7c6d2071b58 |
