# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-084`
- winner mode: `graph_prior_only`
- trace hash: `sha256-57d54d463c3d335756b9ef845ab48b21d6ba79bd455096740f5eec6ab5dcf52e`
- fixture hash: `sha256-c975a7548913ddc09f78bdcf8d6f035b2cb79bee5a8fff204c28b6e92be5b531`
- score hash: `sha256-b04b096de42871946fe5a7047a12994515170f69f7ec7d1971ca5fa398f7fda6`
- bundle hash: `sha256-b1831aef17b14e771a94ca93dd2a99493ba8e9b27b5610b90761c7b6a60d30d2`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6f565862470853eb1b48835f5dc58d5e78705c4b54f6971c4806d12966cc7447 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-be5e0a90759310c4bdc25817cc278d41d19f78a5634989d42b762b3cbc4ccfb6 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-77074e473503573baf89cb2969a0138d0c462680b5b242a50fca7d4f491d1eb8 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-fd934ce9e5ab9348936a76ab28c9c5cf32fb9ebe0c09079f7c31dac8df5550ed |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0b221cca | sha256-57be00cbf594062df7234a445ade9aeb070ac1d757796fa81a5dbab1cb4731b6 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0b221cca | sha256-fbb3f8df1df0346368e83bb9b346718f293a9dcadd9f9ec8c637c099918a8b96 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-96fd6af3 | sha256-d893b22fa71db5106dc47acfaf3c5a32f64c40cc61a90a699bf40af92f99f842 |
