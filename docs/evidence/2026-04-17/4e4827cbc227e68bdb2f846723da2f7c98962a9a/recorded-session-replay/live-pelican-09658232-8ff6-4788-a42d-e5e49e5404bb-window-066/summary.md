# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-066`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6ac1c0fab25c05525176cdfff2149e8d15cf9da0d9dd3e3ff8d1e6b40aadd074`
- fixture hash: `sha256-99a9dda4d1e27d20e5b5802fe99ae2cd9ee98cd875422b1ef45282c42f60a797`
- score hash: `sha256-fb551f8418ec0ba07600b75c3a8cead785aa42036d206ab1fd2af6a8966aa74c`
- bundle hash: `sha256-6a0336ca6db50ca79181a971e499a665e83d4217102c3e484e6c066867966ba1`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-c3402e859f2552a40a7f253ef60215bf90d6f117858139b3ed26992a03a4545a |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-8799698e925325826302ba92931068aae3b8d35fc295899e6a01d5a30a9c2eae |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-491d1afb773ed5ead07b1ee421823c6b559e4ccdb86d7b21d83563ab3827d643 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-4c6fbe66ffabbdec223095ea9e1ee2d2e6e1b1c5c22609cc87fc19d36106f805 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-bf4f2929 | sha256-e8a8411ee9102774ea35d2f7ff227db9fe6fee5364b966bd1784fe9601fe31b9 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-bf4f2929 | sha256-c23dbcabd59e468699e2efc379ac7cfb473cf2d26982d38dddbc856feff0d6e4 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-0f9265f2 | sha256-d9219b26ea7ae0ccd8b5a548510b9c6c5a94edfe30b8a9b22d1df49c36c5613b |
