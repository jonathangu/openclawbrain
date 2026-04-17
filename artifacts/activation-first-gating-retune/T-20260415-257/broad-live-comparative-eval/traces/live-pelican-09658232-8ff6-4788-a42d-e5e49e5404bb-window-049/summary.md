# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-049`
- winner mode: `graph_prior_only`
- trace hash: `sha256-10f30fda1583220ffcb0e13cb73de4976d5f3f5f0f058e8e816ab9eaaeb4bc0c`
- fixture hash: `sha256-81aecda5857d0ab09faf0a56bf49fbe289e64582b0578df3f1535d5bf05ea11e`
- score hash: `sha256-728dac5d783e1d1b5b82252691882c6d382cab6438c6c0d4d5fc2666f923f396`
- bundle hash: `sha256-473bb77f417378b9a70dad8e3b9e12b9da86b806956919d0f55bc0d64be87068`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-dd49736decd703dddc6036cdf0bf744059f6270cb8728fb209a65a281dd21058 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-78e615058f8b408a49562cab20f6cd0b28064bd6644004d33c010ad96cfac562 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c0a961b9ce382258c784c685f5e70ac5f3a6178f3c430a1d4be7b79d166492e9 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-aea0a4acd3d81d278bf0585b01a89b299cc78d110656819cc1a16ce0ad32f23b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-6f29e31d | sha256-c4365fc1e0c8d37b3da8d2902d888d1e1db9ef7bb0dac73466f3bcd6b0f065fc |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-6f29e31d | sha256-a81195bc5da5627fdb0df6a55d015cfa452980e7ddf7a8f0825a763c95d15fee |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-cd53b798 | sha256-7e71372753f34264641ec099252258a405ca3c6e9fc04b6a52b53c4c44ae6b0d |
