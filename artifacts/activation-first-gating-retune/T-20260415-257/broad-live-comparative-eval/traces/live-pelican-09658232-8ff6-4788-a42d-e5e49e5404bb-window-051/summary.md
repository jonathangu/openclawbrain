# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-051`
- winner mode: `graph_prior_only`
- trace hash: `sha256-00bf2bd686f7cfc027d3b7749683ef5ae4ebe1a8b4b5f12763771285b87ec8ab`
- fixture hash: `sha256-1287af06cb4b83146712d66b78f07ce6e6ac74450d156f3cf86e05b95cfe0f1f`
- score hash: `sha256-bb2bc3c371f4bd058e6b4595f931b96e1cb740bc26b7c26ddba6c72747d56be6`
- bundle hash: `sha256-129751824c0aba65b4ab7ac7ba348147ca1d52fb6a80ae6a25ed083059ccce14`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-80a7f9806b34ab4aca7f2c918d805e0ef978c8cb5147a44aad086817dfd7315e |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-29615569e543ef700de080adb653658df0f9c1edcec28a05199956acc31f41fe |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-dacc412fb112ce54072161d03320db2058048d371d3d25ff75478253175bf5a0 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-f408430ddad595a66dc9365ee9725b3350a3f75abf14e7b290c6926b5a5751af |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-2dd8be26 | sha256-996af68b63d93bc84c74fe93f9befe441c8cdd4c6389b51abbecbdc8cea397a3 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-2dd8be26 | sha256-93ef68b169e26f4c69ec95fc8da1a9a320e345cdcf459fd1ed1a9cf7c8ccd5b0 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-0d48d76f | sha256-622ced66fa7d891e786f041ac161e737fc8d89141ec0c546e1ac74fe8b16fef1 |
