# Recorded Session Replay Proof Bundle

- trace id: `live-main-f554f872-80dc-4165-9326-c85c48df2834-window-001`
- winner mode: `graph_prior_only`
- trace hash: `sha256-a9a0d976d691035c87611c3bf8262c8f24aa8cb2b2147e70d168e8ca66af5301`
- fixture hash: `sha256-5890b682144351849269c08cb811bbb472ba0970bc84bb6b237fb1117c406a77`
- score hash: `sha256-c54394de7c9002cf54723440a10d6a06671afccc49c1b993fc60da6fb92b1f05`
- bundle hash: `sha256-ebe777cc720e633831b37464335e0a8ba93e34f532b8ad200aa2bef188bd226c`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1b4a156671a0a476fe0b5ef357aaac56b6741f1cdc3373d320b9eeef3a821f69 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-b4e063bca7120b277adee4080d6b35bfa454655ef46cdaa8d82d6bb97f64934a |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-aac90a97e883d178688a676a9863b8b3a2125893ebd5573c4bc433d750d9dcad |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-ce67e47f9f712fef5609019266ba77427b303817006db92b8642f5a3df93059b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-cf3e03df | sha256-d81fe0337483599ba111011ea68a3c9b2081c6ba84d5b02d1547005e26f2b9a3 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-cf3e03df | sha256-d81fe0337483599ba111011ea68a3c9b2081c6ba84d5b02d1547005e26f2b9a3 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-cf3e03df | sha256-21347c4ec617435731590d243828d5969a2b8c560ef469b464fbf8079bfe2b4b |
