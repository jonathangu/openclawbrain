# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-145`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8588f62e6cb39b6bbebdb00e938513a4cbaa506b41be87532a11b4304976dd66`
- fixture hash: `sha256-523f979d52f465f7796de01a235f1b7bbea1b624b0a2f4aa71ab4b02e1ae0958`
- score hash: `sha256-4e4a21f5b2d313df7f4d5281ce65aafe714dcff08843b4ab9ebb236ee4f54f7b`
- bundle hash: `sha256-4a2437ca3dce56dea5476cecb22fe16d90cd00d9ece08f357cfb04b3e3e1a0ac`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-172463d69f5ae184c08f379b77a680b592819857917ad8f3596af66f22037f0d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cba098bcb161ebe5e35cef47503f8eb24b3fe8b8acda1af55ce718d8e4c4e850 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-70a62b1004844dd160e9bf59b581251a6222cec0c13b06b3cc8bb12490f99f51 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-17985ab1055d3a07b8316cda8e1fd09068f0719f8c71bd06cf61bc03fdc50cc0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5db28b7f | sha256-65021799237c4edb1fe67b81cc881a8921507f1192bf4bbbcae31c61ad58d0cd |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5db28b7f | sha256-1cff87b3418cb7c783dba7e2149bb3f3bde2a7d71a816c549376f5be6a032e50 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-5db28b7f | sha256-65021799237c4edb1fe67b81cc881a8921507f1192bf4bbbcae31c61ad58d0cd |
