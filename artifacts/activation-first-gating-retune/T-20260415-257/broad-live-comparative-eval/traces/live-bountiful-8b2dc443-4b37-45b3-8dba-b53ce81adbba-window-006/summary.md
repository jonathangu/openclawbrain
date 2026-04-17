# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9a9f3db5c5e9f18aad5ca8aa8c8134dfac254479399202badc35306faa348393`
- fixture hash: `sha256-af03997f06ab50c99afcf76923b04c21e1338d145564c582674e59eb816853de`
- score hash: `sha256-20c83a387e5697462a2baaeef76225c94a9780e2ef492cee5c4368e8b0b86df9`
- bundle hash: `sha256-821da39840211e5b13f4180a9c89df06111501da3c59d469686312e3b65fb07c`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-048408a5bc1e1d56a6cc83e227b9a2958b83cb861b21925fd209ce4b8456f636 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3be29aae4fd335c8cb17d8caeddc8ad6db7207ee1af97e4423b5eb7d45dd1289 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1c8be552c69fdcc905a20cfb49e6708e12028735d5b47bb815e723db1c1a627f |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-f6ef9bd76aad25034569721228c86939855172293d4a8210a55d4493f2303708 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d854d6f5 | sha256-4fef842ec48d05e203f45ad15354f97bb05c99e317d69ab79cef495743960ca2 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d854d6f5 | sha256-ec19da7548788e92182924abbe54e8907278fd07732f11f3dbfde62c3923b2fd |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-0c0c2a0a | sha256-c6f76fc5855d1f22cafba3fa857d01dbbc789606e59f4414d1cfd1fb74364dfe |
