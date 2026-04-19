# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-084`
- winner mode: `graph_prior_only`
- trace hash: `sha256-68b9ce7f8532cca72a3aa69a14e2f86116ae9296f4d403b4d6550bfdd087e76e`
- fixture hash: `sha256-9bea38d812adcb691fd84d63e7a18202d43fb587cc03ce111aeaee7b624c5f99`
- score hash: `sha256-2239aa0819cf2804994c6815851152740dcaee4ab4d4a4201f6d9df27b36e870`
- bundle hash: `sha256-10b4fc7708650ffcc10e8f0be8b7399ffb9a59d9f004aa5ac48878b6bf2d45e2`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c60ca09a31056398ac0417340daa846ce032069fff53d8c27abc0aa34201c8e0 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-f8e40ec90c486a756eea4ff96f1a95b87ac11a8780a00068e23091960ac86d9a |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-29f83e72119c51d50c0b4adb72f308c1e427823d48c895007ee1b5d860745e9f |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-d53bda5daefb81eebee57c52606c2431888c0ca0eec1e0bb74fe5cea9ef84bd0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-8119cd17 | sha256-7d5b125f650bc8111cccceb3e3ba66b30b8d875f0963b10c3b914b6a5ca8e02f |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-8119cd17 | sha256-9fa1ffae05c1852aedf4770e750ab68a2d0b255781b8ac500ba71e5855a5b4aa |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-8119cd17 | sha256-7d5b125f650bc8111cccceb3e3ba66b30b8d875f0963b10c3b914b6a5ca8e02f |
