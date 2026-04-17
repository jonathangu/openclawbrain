# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-df424694932b0793aaedff791f54d5ac971c24ed551452ee216f10c505396c8d`
- fixture hash: `sha256-cdd5cd85fb616c8f44b236f115a79978bc2dcad4597a177039207ba517f1bddf`
- score hash: `sha256-8c1467aabc8826b698302f5da3607c9784dafc6df9cb096e1ce09ea0d31d35f8`
- bundle hash: `sha256-9720b9e6ce4c654b3090852adb001a171814dd75e2a7531d7bbb0467b7ca1ec4`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-df3745ac4e10090248775f0174e4f7f9517bcadad1b8588a0276c1d2f867a57c |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-69f051f1935bc7b0176bd71644b3a6a6e77d76cd0e7d250e588c42a74d7de37a |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7662c3e4cc5348ea05068cfc0784afa8d7d489367fab9871d0992ac31b5ba566 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-6a6375627c5665a5a3aa4c91fc893c30d1809295281ee54ec88fe6360e8cacd7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-2668d7cf | sha256-eaca04729b06319bade3bc398aded94dae457ac3785e72822df181ac2d10e08c |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-2668d7cf | sha256-eaca04729b06319bade3bc398aded94dae457ac3785e72822df181ac2d10e08c |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-3197c784 | sha256-de46c77f36ad6b9bdf913c2e26c55d412fb1971067111d40b63a4bf4b7cc167c |
