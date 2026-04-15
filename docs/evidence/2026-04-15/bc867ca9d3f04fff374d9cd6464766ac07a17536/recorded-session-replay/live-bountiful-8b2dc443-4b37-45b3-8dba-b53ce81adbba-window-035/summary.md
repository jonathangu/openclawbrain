# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-035`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9ff7777be0b897266208f103a2bd1fea9aaefd91febf0ea117187545ef2d2014`
- fixture hash: `sha256-7993becb690144ea7d947bba5815a89834c7be01fb7391679807d26712c8efec`
- score hash: `sha256-0d507f250852fc2f9d3936a870d8f6ee156f2d02c524c447fe77059b452eb1dd`
- bundle hash: `sha256-b028b80499ed5c8a6e6969a79afa8880fe7907a4c90fd0c413733b0b9b968fbc`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d3fa2d46cbebd404032c589d360c14fd0cefe70dcd15d65b4e1f8159657f983c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-073c3c3c7e3fc1c6c1a53b8c8aeaae307a716a2fb3d1863293f10a8c589bdba3 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8de4eee83e326661d58845b143dfd114c734679b5984c918119bcdef590f105e |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-f4fdda45a6b6df09f092c70b28f61cd1b47f4e4c19e5e5a4ced23741f03b49fb |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7fd412db | sha256-9d004c5b8376d11c14cccecf3b48c2c37864b9ac2a23f2e12d9f15c2f9f837ef |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7fd412db | sha256-16d37e85d7c40f7887f63514b52e28cada9232f27f4d199a1ef20fca9e92c719 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-7fd412db | sha256-9d004c5b8376d11c14cccecf3b48c2c37864b9ac2a23f2e12d9f15c2f9f837ef |
