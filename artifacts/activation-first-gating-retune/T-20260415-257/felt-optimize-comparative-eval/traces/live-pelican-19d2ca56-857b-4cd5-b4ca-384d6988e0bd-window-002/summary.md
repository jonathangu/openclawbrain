# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-19d2ca56-857b-4cd5-b4ca-384d6988e0bd-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e3587f37a965ca48e3b14fe490f41619f4a64d9248201fd791da49328673f2fd`
- fixture hash: `sha256-72b30c69deb757e882e827610fa9efbae23b0f6f41cd081abb4ab731c8f4dc73`
- score hash: `sha256-f09ad8a0f41261afe248b898ff9ab27d0777d9fa1bfe10655a2c84746c664e1f`
- bundle hash: `sha256-7ced84e1f2766a5483c1ff1278e4546547d6c5ac4f84ee606524ec5b4f3d4772`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9cbe4dccbf152c20742a7a0d9f6d7f345aa7c7916722159d8bcf4f7a084bf5a1 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-942f2dec8ca20b7bd8a47d8df669b9405b3d9937a807f2d0bfcfe91522300dd1 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-af40a732a689a41c110320d84f5473d0dd5f1f0ae868be1eb63fdfa7733ee268 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-1c413bcc123d58e0cc4fb853f87635496d1e4076bc981636dff484964e6176dd |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a42e78fd | sha256-77120c8422faa18db5c5551e4fe080c86b34171c744e62a0c4894242c94481bc |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a42e78fd | sha256-1d216ec377f49d046789b41e766ad9df2a600182c6b650b72006dc02aba7edb3 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-ab59cde2 | sha256-d14561ff841bdb2f658c21f433d6ed49334a3a28761a3cb0409b87ee4389654e |
