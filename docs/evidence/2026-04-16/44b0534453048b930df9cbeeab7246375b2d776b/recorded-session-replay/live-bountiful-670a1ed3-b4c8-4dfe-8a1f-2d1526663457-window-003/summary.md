# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c77161fd107da0850cb368a9be7a432917f75dca7a822871991fd4fdb28ea1b9`
- fixture hash: `sha256-305a6c8327f5df890119bbc3711133fb545cdd219a8e22832dd1a9b40c670ed7`
- score hash: `sha256-f0ee407b20c567749d668b845bacbe759d07c5c45cf0c1bb9b175fafad85bdef`
- bundle hash: `sha256-a3dc4a84a70a5dcef75d0e87e521fa1e4c9e9c1785c987359c6948860f2bc4dc`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f9994dc36549b50fc869a0d07853fce421c050985a49a6ae0b76d1bc12cb356c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0421db844a5bc67f0074be15efbf62a491d89bfc64405d4eccc9091887f9e3b6 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f55ef62a5b6cb40d6f507c646522e7ec6d782432735d5d1360a704ee5967dc17 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-dc7a6689d96363f4ae3a4fbade9aa63cf2d97f588e0c24f81913c55a0d606f99 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f7a57ad7 | sha256-dff321c3dcee5c6c75c4b0f0649d44e715df5bee00e27aca20b48e6b15321afa |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f7a57ad7 | sha256-400ae84c0d5e14a814a3ec720d3ed518138c01ee9369b942946375bd40fd9997 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-450bbe1e | sha256-a3dacc50d356566c889dae1fc8abd2b66c48c26155a823eb8fafc68de413613f |
