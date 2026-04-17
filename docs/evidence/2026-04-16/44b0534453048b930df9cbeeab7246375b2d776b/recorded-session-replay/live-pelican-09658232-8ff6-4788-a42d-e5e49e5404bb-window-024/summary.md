# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-024`
- winner mode: `graph_prior_only`
- trace hash: `sha256-68581f69a97780aac278954522193e99993d4befdc39acceb8ff881974cc0178`
- fixture hash: `sha256-d2931cc864933b7e6af27eb1382872e22dbe9358020b6cefacd8fc78d2489792`
- score hash: `sha256-0ceaab84e7ba5ff0bef8d990b95bbc6a2cceeec7d0d11e056e8eac33126f2907`
- bundle hash: `sha256-b45fbf7cfe72eb9da228e4e03030de98c1a4a87afa1bfd1b378a497b349e8232`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-181208f7b843fa2c39286593bf1b96c7f44d97e1cb317cd9b55efb3be3bcccb4 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-869d32a4f4f62b6ca52efbc3d07c9788d01af61c69169a2c53d4499bcda7e979 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-84ebd1ff668da6626190a1e58099bbc99e7778ff6a99e5599c4f175aa587c865 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-6da5672cd58504e30bafecb7de856277667e528caaf370dc7b61b93cb92956db |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-31a56b85 | sha256-600abede26922543c91d093abdc1f3af150b68a5f1d5f544e2ef118bcba98939 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-31a56b85 | sha256-c5fd987bfbd7a15ad78d92979b75deabaa1557fe7eef1af56c72799f00bd3dbb |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-763c04e6 | sha256-2f4cf5ee2240df8faedc834f4385ca9137687544832ce623b2312715a9d8172b |
