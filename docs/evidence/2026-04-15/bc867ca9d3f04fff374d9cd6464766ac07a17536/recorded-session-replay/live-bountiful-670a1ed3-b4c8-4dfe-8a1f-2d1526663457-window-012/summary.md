# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-14ade459ea986baa6e4e71bbbde0e89dc1fae7980400ac765d36815dff4c4f35`
- fixture hash: `sha256-9c30c978d165bf9a25e14aa9b77d9a12a45f7a9014b4a8204bd05ec1ae139d4a`
- score hash: `sha256-00bd60a6a6c8a3261559a22c754ef4e688162554be1150166d2e1d8e5f0c44ad`
- bundle hash: `sha256-0a7a9a063405f42773b3985e95d53873f5e05c09c0a88852e0754e32bf6b72b0`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-334c135bfa30ec156738872f694abf9297995f829f0e8e1c5041f315be0a98b4 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9fcbfda81e4ac9d38be444ab6e644de7c17d554c1ca71bac4871a8cb0fad68dc |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3e38a554c30498e55fc8878d4a68f8fae2e8a313c4a43bd5545879116869d054 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-5daf1cf44fce53a5f3860f121726011387c17f8e39e9d78c083c03ebf955db03 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-dc08cbf8 | sha256-520056770a6f567a5eaf9acd5a6693a58ca34ba339041104075ee593fabc43fe |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-dc08cbf8 | sha256-f0bf0b3cb1e79e9b4a39385b809afae366720334a6005097faeed074801b6672 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-dc08cbf8 | sha256-520056770a6f567a5eaf9acd5a6693a58ca34ba339041104075ee593fabc43fe |
