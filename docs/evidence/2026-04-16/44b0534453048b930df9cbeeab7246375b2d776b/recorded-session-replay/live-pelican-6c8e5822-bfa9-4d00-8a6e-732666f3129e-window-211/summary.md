# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-211`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c8f3ad7fd7e03d5e6a620d917f9922d423fcf350f433bf42dd24d49c0d04613c`
- fixture hash: `sha256-d090bc75588ff2d651484afffd5d21c674237c8a0eae19ac1a18854f75e95a21`
- score hash: `sha256-c7f8704271244d9a89ff79c82f1cae4ab4b5bad509015a4d8f9005556da35d91`
- bundle hash: `sha256-a13059daa738c1198377dba4cdc4b0401e716df7f3c5e379740d4db0468425df`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a44aad7aa94fbaca9460011cc6ae9061f9cd3a6c6afa137136f8bba1929488be |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-469e3d8975603ea30d241f32121f40fdf6821c101eb738bc3bd502bf45c1376c |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-4a65611c961052965bd331365e57cef831ff3e94f27869d7e1c24aa9cd294b17 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-579135e956d9f9def0a230359c1cad281435f86a0df847a314bdcb3e272e2844 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-1eb8c430 | sha256-c7662c83ce7cb0f46a6abed607dcb5de9522fb7c1a06736e3b6ed61b745b916b |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-1eb8c430 | sha256-097afb872d8c21a2bcfb64231bede9753f0626c27051d558af9f94be03fcef58 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-a4690299 | sha256-32311fd0ad6fdfcff94d2542c15c54041ceb5511f101257806d2afb0fc88c318 |
