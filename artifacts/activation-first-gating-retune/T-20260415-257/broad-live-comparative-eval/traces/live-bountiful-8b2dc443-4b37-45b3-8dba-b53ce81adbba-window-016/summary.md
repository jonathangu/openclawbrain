# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-016`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c817aa28a6ea88ab750b90d075966003c4144ca68cee4de31510afc8940af725`
- fixture hash: `sha256-12c8924300be23df2d629cf06b8bf4e9466d47a9b90ef4b0770c780fb827282c`
- score hash: `sha256-1852c40a7145c0019f777dd4c79f01df7138942fbcb9d99ead104a811340c25a`
- bundle hash: `sha256-384109e1b0dc8f02ec8399c6ac4300e3bcade95431091aac075a59f33f6fb5a2`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a755e36be001d38e08b764d65e8f6dd1b01494428975ffb22d7f3f721a73e79b |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3d84758cf080c3a198b2c6f0956574a51c4e83ab7377936fe58225f4ee250696 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9f3e625863df9c1d553e6d9d9627152f0a75cf542831210f359608c79a5ca4b7 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-4f1a15a306be846e1fc00429ea16cd2f45e20bf978b51cb3f224cce54c85caf7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4397597b | sha256-ac7202e91c2c5f179ae91f22dd247f0fd178f67a641db725210e8f05e0dd5147 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4397597b | sha256-a1fbac44f275df37b0a984182f4eae3d47831edd2d7866e547a04fab8c3188c2 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-a91bfc72 | sha256-07ad3f4078619a1a5ef26786008d395994989222a437b2fa954e3936faa899e3 |
