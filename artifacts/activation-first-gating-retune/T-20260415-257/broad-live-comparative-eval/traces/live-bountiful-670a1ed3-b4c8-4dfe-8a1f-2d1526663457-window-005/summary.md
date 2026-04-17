# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-22f847bbdb3bff7bf5823cbe39964b330b3ee1ba23484549f7f4546fac1981a9`
- fixture hash: `sha256-a7f2ea82d1ad7a3badc44ebc7ebcd547c985d36abe3fcd06170981ec576de057`
- score hash: `sha256-0c07fcd66a8aed31c553fa8eadfc72562c017d9f8f77a53cb08d6519d2de2afc`
- bundle hash: `sha256-5c5223a64196ac3893d989b2f759144e4daebd38e4be46388801c43c7e81ffc9`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9025e023a7eb98100239409dc6df273a8fbdc8529118429bd0cb2b4995877ef2 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c522046241b05c913dea34119cbfd0cef756e864d0c9cc4f86df5c88acbbef92 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1c95a21d093760f0c9eeb9abaebe36eadfd9b3a15b5d4f9ebf1a1420e4d7827e |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-016b8e3f828e6c30b755d79ad2e7f674d11a1161a242353958a00fac8c3fbc0e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a1126035 | sha256-378793be772ae39a8b335547edab22067ae3af0240b97b088165260ebab5e466 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a1126035 | sha256-bbd2284626cc5bc15a73d6d65be64dcc311c35475d4246f10ef0ff5bf7954d92 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-4d1c7e02 | sha256-d50e6e73bf3efd7484b87422e1416a12ff451642fa97b4cb84ba99005cc77013 |
