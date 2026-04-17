# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-044`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0ad01cba10197b8b05f27f973f11a9d5ae36ed08ae5a28bf29003ec749435fdb`
- fixture hash: `sha256-6830cd222cfba386b49de3a4d46620d84d8c333ee746eafd9c6c6f8ee2dfa95c`
- score hash: `sha256-01cb1214e0f6a8bb267afc83927fcf685ff0b4b3b4f71d1d9a9dcc5a38ef93d1`
- bundle hash: `sha256-2273f36473aa3d5f4166ff370a690e33f1cad83d8cfc2ad9547bbfb9f0b9959c`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-490c3e2fb439b8787ca9fe3cb573a9c587cc3abe414e2faf637ea2b84d91d268 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-72d56eb8da9ea6d0c631b8ab921e23a11e809e93cbbf2d1d4241ae84d47cb9c2 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b14f56060cf666b43f4500f6266d2d57724de596dfeb01c36b1486a612f769f0 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-8cce3ed1cb702d02a5d2bd60a723168efd50f6a92cf0b552a2678fcee204b72b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-3079d7a1 | sha256-5b58ab7760f2f8ba469c54f135d4fb77cf002cf4b3a5aec5109568a70126a8ee |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-3079d7a1 | sha256-f556799600f5c46677529b87ad91a28a12c2159dc45f5b61c9dac32102df35e6 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-510a7b6a | sha256-d48fa8b9ed6ff3ff6e56a7efdc49f07b97d53e6058a5b874d87b8877bb7688d3 |
