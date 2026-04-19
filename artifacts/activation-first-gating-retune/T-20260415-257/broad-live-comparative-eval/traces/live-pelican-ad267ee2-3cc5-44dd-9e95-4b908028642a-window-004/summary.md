# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1382603626a2aff7d92c871c45318305722032a646fc01502f912f8472d0ed38`
- fixture hash: `sha256-ee8d3f8c272648220db4d9e69e984cdcf85084bd085927ab6802512d77922517`
- score hash: `sha256-6270ce9ea4dad18fb9b9312a777211480d31f8490935972084b8ba6aacbc5190`
- bundle hash: `sha256-62f2621bb0a1bd4f4858b284489ce1cd8d1e82a1770143c58a7e09e7cdf7d069`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-394d692f2aa5412e9da10dfc0baf182beb2043f517fb99b07451a27af9201624 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-06d251a689a156f7601bcad2f95d3eb3ec498413294615c1121f602e862d4883 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-d34f68a1c2f5f98277104349767a71298478a20c8ef5e2e929050c5927340f75 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-c71bc31513405bdad00bb038dbf97f7b6fb8ae95bb007954254b017f4e35e8fb |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-1e02f0ff | sha256-2cd572a9d5430a4d5e7e1808860c6f02e86799383f4f2beed9d3b1041514724f |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-1e02f0ff | sha256-0c71a6515a6701f03655d7d215be8bc23f7d714acbf3f2da3ac4911e89a31201 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-1e02f0ff | sha256-2cd572a9d5430a4d5e7e1808860c6f02e86799383f4f2beed9d3b1041514724f |
