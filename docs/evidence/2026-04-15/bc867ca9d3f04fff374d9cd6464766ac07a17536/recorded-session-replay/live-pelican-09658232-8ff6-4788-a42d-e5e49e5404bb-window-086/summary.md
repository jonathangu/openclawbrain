# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-086`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c70b7e6acafa9f174da3df163120ba16044bc767e199909b1a7b96f75ed37549`
- fixture hash: `sha256-bf91f869d3956bf5fde31cf4fcbfa13c4356f4c344c72e681c59e051bd04b628`
- score hash: `sha256-b9d01f1b5bef4e0aabf69b33f9a2cce896dccdf201b648d2fba18d46d70fdcf4`
- bundle hash: `sha256-6585b9cc278db60dc6b45b80286c95f0667b2a7f8768c218bb43be6e6b20977b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0b139f94f37d6885531ef5b31e5bde18e900dc87fd64f0c8059b9943917b139d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-df5278fb6eeed7d8b6b437c1910411884f95557c66a6e43f37b35e019adb737a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8b66684353e117f1e931703d3e6b19d8a02feaf93802219697e7032027e5520f |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-4f343be6a927c0f0fdaa179df4fde77340e6a335b75b5c90f199bc673caa6c45 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e6fcddc7 | sha256-68a41fbcace5d7f4723facab02375cb5b08383da4b83b5d183370966b88aac69 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e6fcddc7 | sha256-f1fb2c184ddf408f6f7463eee9bc582af5142c06d3ccd0405faf664257084f25 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-e6fcddc7 | sha256-68a41fbcace5d7f4723facab02375cb5b08383da4b83b5d183370966b88aac69 |
