# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-ab517e57-6c7d-4bcd-bce3-265ea08c9853-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9a8a2e6a63cc5912fb58030e76267c771c6d07671775935e13384022cf8e7c59`
- fixture hash: `sha256-d3b9199b3d1fba06ec6d727611496f93d92d13e1e28ef25defc3314d0f80c421`
- score hash: `sha256-14dcca9716ae67b4f9d500fcead533b86f5c36d3c19208ff6c4d807fd6ce17b3`
- bundle hash: `sha256-b4cadf8e009f20a3cb0586f546f91c5238d0cfbbd91751adab38cb7bd5713d1d`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-39cca038bdbd32b11125d0c6fba3b1b3a673e66a982ba05e8a320b541d748401 |
| vector_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-789e2018abbbc880b0c5e8724091bea019d7d7979b4b665718e5c8749a3196e8 |
| graph_prior_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-1d793a28da1f9fcedd65419824a19c239a47d7b9f6422e312e5dca9648ee3315 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-573e0db297d2f9df1a7e678558e1d2101a35201cedd745b1e0efd8754febbb86 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | no | no | pack-43eaaeb6 | sha256-8c17018da1a795a357ac2223f7e7c0780527f7590a67ac1e4458f5604da9e8ce |
| graph_prior_only | turn-1 | 70 | yes | 1/2 | no | no | pack-43eaaeb6 | sha256-1c3793437b5dcb9b37a1e979442b981ccffc3b605c0bc4c6b681313d7b41ef62 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-2a0dc4c3 | sha256-ac9e77a89824b2efad7d016eef4d2071d36692ee8e701f130592747d310f8dd8 |
