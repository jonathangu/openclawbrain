# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1382603626a2aff7d92c871c45318305722032a646fc01502f912f8472d0ed38`
- fixture hash: `sha256-ee8d3f8c272648220db4d9e69e984cdcf85084bd085927ab6802512d77922517`
- score hash: `sha256-5dd98ae037d8fd45eb3cf0459b3592507a6d233218efb7f668c131158d9f9b89`
- bundle hash: `sha256-91e6f69182d4f7d3105bc8e8d0f5a61ddbf40c51ac5c413981ecfe2dfcf69914`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-394d692f2aa5412e9da10dfc0baf182beb2043f517fb99b07451a27af9201624 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7f5b220c294118f6046737f5aad98631995a41be15884b4d96080152484892dd |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-129470872da982bed30f575f71f07d5bab269af062bfc37f60d2490d3fb166b4 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-037cb272a2f1f888332b44b5c1d3d4a1f675cf1cae1a24113d04836343563924 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b753e39e | sha256-9f3d1c04e8e54b1c1438c8828250370a22d06af0e649c1bda9193f039520ee33 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b753e39e | sha256-8500d28165b9874d2d31318023189330dfe8ed2ba419dadf033ed3863700daf5 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-b753e39e | sha256-9f3d1c04e8e54b1c1438c8828250370a22d06af0e649c1bda9193f039520ee33 |
