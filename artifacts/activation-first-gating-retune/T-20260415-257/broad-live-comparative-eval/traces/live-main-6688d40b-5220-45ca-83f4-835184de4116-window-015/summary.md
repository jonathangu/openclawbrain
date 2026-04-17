# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-015`
- winner mode: `graph_prior_only`
- trace hash: `sha256-04fe85c8a179be229d8c68dec97a25113ffce0ef409792233e0ccc1c65106721`
- fixture hash: `sha256-66a77cd573b5398a7b3b4867686fe20ef718501f851c3ff410c457c68968fa97`
- score hash: `sha256-2d87166b34df70070c334fd55e38cbe131497d9eeed9d97a1851cbdf89d3ff85`
- bundle hash: `sha256-a09dafb532fea4fa5186977c306f9eaa464fa52b7c4e90c6bc8e4c15e65e12e2`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ae930e12c21c7056f67f547427d9cdedef7d7970b442aa81b3fdb75182425c80 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b3e62df4e149b4995bb4715f47f95d415c270170f16eb0d03655d76745790dc0 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7d3417c71556f8b794475955391e6592571ec3f24b1345f324e365debdedd715 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-6018449e219090622e13b07a05af0b965da11a8166e217846f44719d5c96b7de |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-78f9ba0e | sha256-10fae66db1bd95fdcfe9ab01060663a06b06a4818894bd374e747f83ffefaaf2 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-78f9ba0e | sha256-7ff4596f8cc9bc3cdc15d775bcd7f52c3bc9a01dc7c84f64452240201d182450 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-d676e99d | sha256-0bf65ddde4dba9878569d823ea94736b28a6fae42f6d4207279a500036195b1b |
