# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-086`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c70b7e6acafa9f174da3df163120ba16044bc767e199909b1a7b96f75ed37549`
- fixture hash: `sha256-bf91f869d3956bf5fde31cf4fcbfa13c4356f4c344c72e681c59e051bd04b628`
- score hash: `sha256-1ec1d31c82e4f167561fc747fda20df78b99616d0b91919a4b79a551ad86d07f`
- bundle hash: `sha256-6af9d482125ba8ddc03f52df0279f50fa52ed2cd0b0e3d082808ccd31e6f4237`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0b139f94f37d6885531ef5b31e5bde18e900dc87fd64f0c8059b9943917b139d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e7c58bdc6210bfd1351f7f9739d3b6366deb31221df032a6397a7c92c63750f6 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-899eb6e3983972d8e469d867c53ccbb47b5335f0237e1d816046ad62f64ce8ef |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-4eff06f705562659aa88710c45bc687f7a95eca0a76ce6cb602500b44e1b23c3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9131cdd2 | sha256-13f33b6417062490f96f8584dbccd19c8f3bc07cc665847d0e657a0ee0bef8ba |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9131cdd2 | sha256-7f5a71bddccd59f128f235236659dda85de1fe436b1801eecda2ac6dd02d7c9f |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-6b2082d5 | sha256-0baebb571726d5c33bd825ea99f41aca3ebbf4d417a72ccdbed4a2b95a87ea1d |
