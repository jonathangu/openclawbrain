# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-035`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9ff7777be0b897266208f103a2bd1fea9aaefd91febf0ea117187545ef2d2014`
- fixture hash: `sha256-7993becb690144ea7d947bba5815a89834c7be01fb7391679807d26712c8efec`
- score hash: `sha256-c7fa2bc25c284c1a2a2d64ffc998d1e54f52f74fac062aa80794ed79af764ecb`
- bundle hash: `sha256-11445ee0491755a6814c696f79cb0fbca79ff201e65fa48a8fc011f8b0d5f713`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d3fa2d46cbebd404032c589d360c14fd0cefe70dcd15d65b4e1f8159657f983c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-da60d69cd4d8bb5350ed9dd74c431d37c2d98848171817ed1fc3d7f77d25011a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8958efa57cb4463936cdb46cee85a386027863b86b316f15f334817017719dd0 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-d5717366539acaac0f111994c67caca736f997ecbe1f5f6558c8a1ea44d5cc22 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-87e1de46 | sha256-3999793c53bde2188b3945335f9928b93ef784b7176fad1bed0995ba5514b582 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-87e1de46 | sha256-585947697559471bec09447c9bdb92285d6a9158ea8d2f5f3792f6645fb716e2 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-d33ef031 | sha256-c72174aaf496c74b7a50676a51c1100c1de14adcd6d00c5cd86219447af0a45f |
