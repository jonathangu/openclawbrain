# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f6225db554a4364f98319fe1020858f0210ac404609a5303b38b0c9b7d31d658`
- fixture hash: `sha256-ed619df46c192591981773c10de0d36b74c9e5dca7f78e3181e7b1aa8b066c66`
- score hash: `sha256-6539a03dd4241c5f7ce44d3ddd629ca805d0bf8f90f2e4ff6537c32bbe2fae72`
- bundle hash: `sha256-b2651e71a7a8e97abba59aa72a6014067ad8a6313199d623d1684ad4a572d21e`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-88c859d8e45ae0f9acc890628cb16be0104a115fdd6d7748c85da1fbfab29bae |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c9b821470239ec42f269ceb250ff7e37ce86c7b7e9b8aa1e0b6ed662297d13ce |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fb8dd518af385250258a95ead8f4d0d5b4471044b07fff4567d2ec0ac89ecbbb |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-3fe233181d9675f85f7c757908b8e8f0386a5add87b569cd54e845fb2c357932 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-66cf9d8f | sha256-93155455d6a4af91ace5a0e6018b1ba4d4a7dc8d9761ef1159b6336c46472f85 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-66cf9d8f | sha256-93155455d6a4af91ace5a0e6018b1ba4d4a7dc8d9761ef1159b6336c46472f85 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-66cf9d8f | sha256-93155455d6a4af91ace5a0e6018b1ba4d4a7dc8d9761ef1159b6336c46472f85 |
