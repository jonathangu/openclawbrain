# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-041`
- winner mode: `graph_prior_only`
- trace hash: `sha256-76273bf572d2d6df7f5708306c7a325e0d7fc022256a8c48664d2d2c99f93d6b`
- fixture hash: `sha256-5c64fa2fb4319875db5a6403e087c56d2c1a468e1ff9a819a4e71ac1b0668ff8`
- score hash: `sha256-0f5f630ff54b470c9a38d0f85f039ab9ed04261e1cba8482fe333fd6bea590f3`
- bundle hash: `sha256-f70f7e0adeb25a487b0cda98cb665d8fc56e64ca9c027326e926db79d7ddf1b0`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-dd39b973453e541375d3824b8f2b46f3993347e9ee385b937ee54648b5838113 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1b4a5c7885397d91c17759fff2107dccd8e3a1c06f4a4f1a79b3759931616405 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c8181bd4be0de74b921b96ce15e93ddd27905adbc25006125815df69f0fa89e7 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-6daa143750ad5b50b94779b2f75723cdeb376bec723d4798553191fb72c18c5f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a1cbf8b7 | sha256-168679d27036cdb082659d861263d5cf1c131d1f429b934ab4e54aa5c422f748 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a1cbf8b7 | sha256-168679d27036cdb082659d861263d5cf1c131d1f429b934ab4e54aa5c422f748 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-37f78ef4 | sha256-bad39facb3c2da9913778d9a48811d5cd31aaeb7ae07b46d4e5721ef92d7299a |
