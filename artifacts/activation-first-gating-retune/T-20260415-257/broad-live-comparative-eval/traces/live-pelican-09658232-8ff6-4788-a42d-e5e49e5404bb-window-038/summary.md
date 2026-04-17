# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-038`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7fb85648e6a75ae5a3ba6cc73943d9146864f7df475704ace40a5204fa142526`
- fixture hash: `sha256-7f13329cf1857fada1958a6fb5e614617a7842e6a6185ecb6a9d160264a3397f`
- score hash: `sha256-549dfc413095e2b9031ed5a28843a28357f06f852756ab44a5fb204189a1e6d5`
- bundle hash: `sha256-2402d5bff8e9323e285d77f92ad116d4b710fd62294b97b6720d5905ab71472e`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b8dfb1b133ecf1f6249a4f0b0aded2fd5af80368793a13e3eb702ebb8c1e8fca |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f214685bfdb5c40e9de77506cf6499e38f8d44f9df963fc9d33a03059dc3f0e7 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3b9eb34485a1d46445030f002a2170cfd8897f6937e40055a57913ef7a141405 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-263a217cedef4d0b32da75293159294722883413bbd2c40375436567229dc283 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e60d8684 | sha256-ba8c71d0dee5037a69f348d1c6059c94f705c8db72cbff5cb1da322c6dfb2861 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e60d8684 | sha256-6375daa944148d4e58357d496651273d852a861b292f3b145554b12e442fc507 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-1db4d6cd | sha256-5fdc1712d5c241432790624d2fbf937c8d277795a5fbc23c0ddb4fb5962e0e41 |
