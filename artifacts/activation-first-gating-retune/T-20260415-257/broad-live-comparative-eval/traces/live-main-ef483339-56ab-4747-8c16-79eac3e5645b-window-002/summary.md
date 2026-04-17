# Recorded Session Replay Proof Bundle

- trace id: `live-main-ef483339-56ab-4747-8c16-79eac3e5645b-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5348a011d171022e0b0662292622dd790b7dccb6110063ebe79c7f32c96cfe4b`
- fixture hash: `sha256-076839ee1f2768f5fb0e1a395f80dc28e7868b4aab96a489d4fbcd347a8fc395`
- score hash: `sha256-235ac30766fcde76643acef2cf625ccd21fae6d2c86f28f41ecbe00c9969b536`
- bundle hash: `sha256-d4e6645065f8431ebbaf2409b49203d62c2500ec8241264051b1d435e166884b`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | vector_only | 60 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/12
- phrase hit rate: 0.166667

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-81e33f098e21a8124ae6ec9568c1d72b0f83fe94e5b59e948eed0392a9dc9438 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-8d207bab3298d02dcf6f83cc7e350fc402fd2bd3ccb0be4007c5dd5690c820d4 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-4587edc7859ee7419b45a17cfab31b99cc361508eb0991d569738105d4413396 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-faf08c18046fe2e79e382586f2396e2c606643fd91703ae55d53ee652ca43274 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-38ee0926 | sha256-c6fc44566738accb849a57b79e6779aa47103d46f95df9fd887ac16553703483 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-38ee0926 | sha256-72817ecb7925d6f06e6af64f1cad9218d939dc148f51495150a5db17a193790d |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-77ae95c7 | sha256-3900bff63e0f6c355f2b85ed6cac30e4a87535493b5a370149cf2c6f9210ca7f |
