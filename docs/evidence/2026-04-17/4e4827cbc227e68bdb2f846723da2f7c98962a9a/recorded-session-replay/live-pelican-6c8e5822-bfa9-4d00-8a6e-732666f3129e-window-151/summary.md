# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-151`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c1b296177f077b6c8091fca65eb450be8d5f631873f87466a2fe9011d8b7c085`
- fixture hash: `sha256-56e21e2f0877b996d5170fefdce01e8f6c2815e782b17ac6f82fa56c1dd0500c`
- score hash: `sha256-3bfd68bacf19cfdeaf2be8baacc7354bbaa812b6db3842acbc9687d6ef2c29a9`
- bundle hash: `sha256-cb435add528954c1ba96d234cbb028faafe1e52c15c3e995dcb00cb4a7d9151f`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-29c5109446f744c540ec8fb2d0eb8a2d5f87ccaaa85851914bbf19fee8f8ade5 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-aaf7668843c472ad98fb94e1dcd172c8259901953e89a979590eb3283e3e8d37 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-87adc5b37c7eef23b482aff500c7ec795f0170132b3142b49e2de658a6f03680 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-2805b5d8b6bc39ec642672beddaac2b590a327571dcec5084aa6d2774ae68120 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9f484c9c | sha256-40ec0b82b8a3d9523b2497bbf5c1e82f51f1bd6e0fefc03d85b9bb0c8f52246e |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9f484c9c | sha256-0dfe0366a3d3808cf731f1a802313468b3f3dfe04797edc8fd540f083240576c |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-66480995 | sha256-08a8fc45c91171ac4bfbe5ce60981aa9716bfc0959c84f3ccf5953195db9bd65 |
