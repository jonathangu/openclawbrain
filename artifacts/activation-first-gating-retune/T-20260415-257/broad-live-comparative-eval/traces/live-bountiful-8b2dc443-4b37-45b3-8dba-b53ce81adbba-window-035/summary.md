# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-035`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9ff7777be0b897266208f103a2bd1fea9aaefd91febf0ea117187545ef2d2014`
- fixture hash: `sha256-7993becb690144ea7d947bba5815a89834c7be01fb7391679807d26712c8efec`
- score hash: `sha256-b575f97900c5043c6ab71adc969f4b9a73d4c3d2a8a29e9d3d93f7c855a32f20`
- bundle hash: `sha256-1bfec5b44df977a7d5e1d1c924cc9952bddecdde8bd2d2df6c3017c4a0a9b5f7`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d3fa2d46cbebd404032c589d360c14fd0cefe70dcd15d65b4e1f8159657f983c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9c7c78918b3e293812fae4bf0b196e496d49ee92430f9fd79412568c6964222d |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-da0378229418f580ce07f16dd2a2254b543a2a30f829239322eab8f21b76e2db |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-7d836a99fc5404d43b46eb1d24e1edcfbe55aacc8eda95c49805de8a89d29398 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-52141bdd | sha256-0cb5e2574f3eee8b64063d18ca45ba21f83c6f508579f73bf2877af07d696ff9 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-52141bdd | sha256-9cb1f9a95587b45b05c38a07cecd34a02751415e7f9238f92d5d2b826f612b6b |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-9d712dc8 | sha256-7b15fef283e297a8c32be7768e87006cdd60607d634e1bff7b67ea8e3da0bab7 |
