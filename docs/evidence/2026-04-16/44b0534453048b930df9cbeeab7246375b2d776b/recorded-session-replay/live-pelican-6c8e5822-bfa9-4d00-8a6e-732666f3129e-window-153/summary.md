# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-153`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ffcf94e58297f053bce53168278403d4ee13aef69fa248575deb3926c6117a0c`
- fixture hash: `sha256-69a203d1bba5e9efdb04c3d2b5eac78a0fd9782e268e61f935bcf93878b096ff`
- score hash: `sha256-6a1c1aceed405ea91e033c26a5b3346402e7b93013ef15d575d7a96cdb99fef0`
- bundle hash: `sha256-de0a81f6cdd7b1160a5b20d9d4ea38d279eebf92da3891d80caa64b50001d1c3`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d38606df0a6b5cc6fe27f296186c09efc80579f0832811cf6184d8073ca5500a |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7e5170a4f8924a1d903af6af467787a52583073042715b515388488ead1249b7 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-dd6881a78636a8474b0e88e17e7f7f3361c58eef0457da251f143d89aecafc80 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-99df1c769d2adf76fee3b4ca22b84c4e87c9a692081e33abb048f718bc9e6cf5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-54529800 | sha256-a3ab36a71a09258680a61332f9fd5348675fe0bdee7e7623560bd593cb8aa68e |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-54529800 | sha256-2247ef53f975c42929e37151504b2f45273b0a10be541d31cc8582b1245c1099 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-5d097861 | sha256-07803989a8267b4474c396402cd62eca555a57e4ae1d4583787fb37fd3abbec6 |
