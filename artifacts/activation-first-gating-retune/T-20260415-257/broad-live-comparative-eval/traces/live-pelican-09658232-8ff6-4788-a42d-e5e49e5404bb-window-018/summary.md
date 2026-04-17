# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-018`
- winner mode: `graph_prior_only`
- trace hash: `sha256-31afb2df9c1a17ca25197bd8dab4006e37b9c5d5cee2757703f3aa5a6af3cc63`
- fixture hash: `sha256-998f6618e36f06829cb18a9eae15dbb334b923e47c420cfa28a2642db4d68155`
- score hash: `sha256-1a292842b8970a3035040deb0c76d55f2999e507778fdced66626f8e93469da5`
- bundle hash: `sha256-2c81f32c85939b6a7bfd9150798e924bfa7f754ee609d81dac02e160ca5dc725`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c0657d7b41d16e0a76bed7b5e5dcdadf4310444b0556eb5e7411f6141dac5dd0 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c70d3d040f4d84e48a32711e3557f4a21241cca986174bbc575f1425038195ce |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ac297731a373d741031f1d827a1239abffb4abd1985bdcdf80bd1bb8d85b533d |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-796cc7e4bb0ed9b99ea5f9d4f94ed331e4091748207aa008eb1dc6e53cc7121c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-016adbac | sha256-a8076d5c3b8e5c4faf44380ac95d99a8ddfbe765e920c0e69c97f4d7cfa89640 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-016adbac | sha256-a74581fa868b1a02fef14046a836b376dcf3b7a1a833f0c4cd3ccc0f65c5fdf1 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-9309f853 | sha256-ecc7710b69ad760cfd316fe9fbea80142fb680ffab69ba0dca848e455bbe69e2 |
