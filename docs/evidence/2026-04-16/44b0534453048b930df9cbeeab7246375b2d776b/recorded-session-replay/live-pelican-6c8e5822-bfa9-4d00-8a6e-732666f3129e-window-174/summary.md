# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-174`
- winner mode: `graph_prior_only`
- trace hash: `sha256-681abae2fa5f82c72e9292e394b227021bc61d412148159906a6b997f617cca5`
- fixture hash: `sha256-c46f6cfeb0331761c1d2bb543d4b028a9a876c69162435d955a285bd82156828`
- score hash: `sha256-8f3bbdbf0aa85d7b52f03eb12072959c4d143b253fcad525f569b7f81902efeb`
- bundle hash: `sha256-6022d719dbed6e7b267843ca00309ef7e9930036573a0555c7b98c440f194a3a`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-461e5f784d6e942b4cdd1338a01f757f830996458c9f4abe17a0effaceafc63b |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b16235dc6b5ae0e8443401a3631df4208c827bcd47a41784423de4e3e92ce96e |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-064d7cbfd54cda3733866e5ca8378706353fcb5f089452c81f3a621aa7fc1d5f |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-a3f3c35b37ef75a56a4566a64e3f3854c395a3265e173d61d051f4b82db5213a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9aaff370 | sha256-d95f2de4f63e3fbc45fc73d33efccb6c4a303dfc4bb0c050915df6beaf6d998d |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9aaff370 | sha256-e0132af1f235377e7a0536337aa0263c743ed6b13db240632b94c8376587133a |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-5872c2ed | sha256-30c005c18170abc9fb3acac0bfac4dac6f3e23d55d9aa4015af8b966781bc6f7 |
