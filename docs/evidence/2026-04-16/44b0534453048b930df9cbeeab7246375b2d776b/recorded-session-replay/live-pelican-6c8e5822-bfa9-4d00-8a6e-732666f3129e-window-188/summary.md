# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-188`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2286f1962f858995a9d11d68ccd4ff744be8c0925ede8b9595870bdf0f8216d1`
- fixture hash: `sha256-7131fce3dd7f89b87927812976c9719dadea253e34115e7f37e0887827e9427e`
- score hash: `sha256-2d9c7de756b209527d7c02c0051dc3c41f5714cb08d9339625fa4994eee2591c`
- bundle hash: `sha256-13851bc8c4302459e45738a7e0234fbed523131d048abc93ef3c7105cda8be68`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4cce362e58248df18caee133abbb86ea37c7c8cc312d9027b572d5a719da7a87 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8632f218ebdd7b287d78888db92064d998291b172c563379daaf0f203df9a86c |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ae6d75b6118838f24fae0ee3af6534534cc6961112af004a778d14fdc8ccc4bc |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-8d328e8bcc4f02376b823522299a8cf02d7aedcb740bb3a97ce9e5c48fb10835 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ae65ba4c | sha256-d3a6a7bee1353c3c7415bfaf36846354c5a1739c4b488e260032e108f6b2a30e |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ae65ba4c | sha256-ef866848d4e05d4bb299741d4acd39b3c6354bea8f309711a63720404612e652 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-81c22647 | sha256-a09da743ef63d2364800eecfe8e77444a37fe84135de4aea4f65581dfc154ccb |
