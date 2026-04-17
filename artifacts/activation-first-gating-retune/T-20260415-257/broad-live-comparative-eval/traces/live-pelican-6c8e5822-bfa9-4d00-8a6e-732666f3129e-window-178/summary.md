# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-178`
- winner mode: `graph_prior_only`
- trace hash: `sha256-dc837ac64ce4a5cb1d121e2bea7830254f5b1cd1faf9dd8be0505cf94fe18342`
- fixture hash: `sha256-555eb18092c7a3b48bf36359187522f84e12b063bd73ce65d859cb8f468c2af9`
- score hash: `sha256-af8fb41cb44e12b846968826899b1fe2c88720ce67fe55be99727013f30e5ed5`
- bundle hash: `sha256-c29d063fe2ef7acfbd5f0d1708255a329f025055f2e75beece4f639848dd52c1`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-94018db213a88670c23984311d9a8431beabced6aba3b25434ee10a70b79887e |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-51bfb854c907c51a09fd974ff4eadd2f1f139221c97d5217c0fd058b9346d62d |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4ff828f5680ed91e929c2281b1b97efae44a6ed4504f79f9233caaee1729736b |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-f10acad07b8f553d715c1b5c1bcfa84a63fa205c8f8014fb87d61ccfd67582bf |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-32c06c75 | sha256-779caa5ad634fc4a3fff4dafc07386a7438fd9dd792cd459cfba867575b76175 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-32c06c75 | sha256-580acf497e41c0e346615a579bebb8eb5703bd1fde82288be3b09a5f096f6961 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-cf72f21a | sha256-13e8ef3f28b48fa90364c80f78fea1e71023f490e469a99524bee41171ab5608 |
