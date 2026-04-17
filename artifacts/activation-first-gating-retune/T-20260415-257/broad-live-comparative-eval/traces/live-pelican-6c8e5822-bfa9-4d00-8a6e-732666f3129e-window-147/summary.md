# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-147`
- winner mode: `graph_prior_only`
- trace hash: `sha256-12b53203712e88b756dee356041b3ddb0e18e328e1c8f8ade691064553599eca`
- fixture hash: `sha256-8ac6a4fe3950f0ed5cfb2e1b9bd9c7ad4d79faf9e22bb913250d8fa59920cf2e`
- score hash: `sha256-b40e079f73ee01168b02f7481c1d7ff7cbe7e95658e2627dcdc74fcb11ec054d`
- bundle hash: `sha256-1d643263606aa7c01c1a867eee72d4d15a973db98fd633940eb68f48a9ee9764`

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
- phrase hits: 0/12
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-55766afad53c9e202670418bdf755c0f71228a26fa5f954c36b74006ec3fe092 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-792e28f49e87681e924a52b8634385d7cc9904f1174c39104ef1759d51de9e9e |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c755f018db4ee22f0d9e09f1b715013694cb9b77a74e985ce7cdddbca4c201d2 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-f992a8cf4c2e8a478f33ab2ddfd8906e3fe42d098410348af1593483767f4298 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d4bfaccd | sha256-6eed44b0ec46b4aa6eb39ddded23d5c7eaa93d6d4f209e7a59b40a47e803dbea |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d4bfaccd | sha256-e72aa806d1f8fa091e099cb7efb1c317b20ad8448834c13c2a535e4d7abdda4e |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-d86c950a | sha256-7d493b5b408bd301fcbb074efa4e4b710f3208411974bb9d265d8008a1806f0a |
