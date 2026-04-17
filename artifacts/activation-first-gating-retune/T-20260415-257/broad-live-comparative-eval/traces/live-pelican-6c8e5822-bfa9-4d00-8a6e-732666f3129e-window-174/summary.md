# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-174`
- winner mode: `graph_prior_only`
- trace hash: `sha256-681abae2fa5f82c72e9292e394b227021bc61d412148159906a6b997f617cca5`
- fixture hash: `sha256-c46f6cfeb0331761c1d2bb543d4b028a9a876c69162435d955a285bd82156828`
- score hash: `sha256-025c29a5be666c19b4fc3637560dac326b4f9257a53e05b141f6ebd3024c373a`
- bundle hash: `sha256-142d22ed104827ac8d6a7a989373ed6695e9642829cb32bad2aec39c577fd123`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-461e5f784d6e942b4cdd1338a01f757f830996458c9f4abe17a0effaceafc63b |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ba25745916f73514a9d0720e4f66230c552934e900bc42d05da62de204b37fec |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7aa1ad3803e3f232d7fa47eaf1443e4f4908a1a7e3faa5d72e52e3177394d3f3 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-e40bee6b278277a0dce62aafed66c1c919e63526458a1ec2e307cfdff6be8da4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f4dd6032 | sha256-fbd5efd1f626f92f5247c958a2707922d39c803f9878635fa20d477e2f830526 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f4dd6032 | sha256-2220a6d0b8dbe97d1d48ff4b66c7fc329134bd2b7af095044c6a6734c113e4c2 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-b2a02faf | sha256-0cd6bea1fae3c588d3b6185cf750c43a9019dbce704850a79433c2825c481e3c |
