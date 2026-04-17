# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c77161fd107da0850cb368a9be7a432917f75dca7a822871991fd4fdb28ea1b9`
- fixture hash: `sha256-305a6c8327f5df890119bbc3711133fb545cdd219a8e22832dd1a9b40c670ed7`
- score hash: `sha256-9a382ec1c0f981a1c4c15e67f9b41ef63dbd5cdbe2e7b6049cb5537a7717c9d6`
- bundle hash: `sha256-178532c5242e331591be5d882d2b2cbd4a25e051aa8f1a6e49049770d2e1c824`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f9994dc36549b50fc869a0d07853fce421c050985a49a6ae0b76d1bc12cb356c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3e73e833fde1572749ebb09870d68cca399f8555bd3e7e3fd271d3a8d307bd18 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-980e78ae362a95836c5ea7527aac07ee1bfd3102be45b4d20bf7a10546e74a9e |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-1a6d7decb79e5fea858f9d604e03616d27820fd46143359b063e72efccc98324 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-15cf9c65 | sha256-8d10eafd44424df78e36274d5692697061d4cc4778c01cf6485f41dcb6105524 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-15cf9c65 | sha256-1033486a015e37336424ae5fb417bf1c54fe4c1cf74414409ddc8e001baecad6 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-6335dfac | sha256-b407ea0017dcca299827ffa4220d36a7e06006fae6eec1b531a205abf8718485 |
