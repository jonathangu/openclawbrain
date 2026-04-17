# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-dc7aa6c27637d299d6eae706b4fc67a2a2a7b4de77c818a562317ac57ca7ac6f`
- fixture hash: `sha256-3d9a8c7638fdfa743ac7a63700e6bcceed5b6728eed1bfa78f1b2db0ab28c6de`
- score hash: `sha256-e5525a064110fa5161918466897cff80a105de094bb6e66c68237c62c1a541f9`
- bundle hash: `sha256-b8dafe73ac5a7c0372124fc34a690f85f1b66bba24ff9a6fff32a48742b169a1`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6474375f5bcb6a5860753785382ca496af4bf19e7ca31262302583c0776eda20 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-efe8881d0fb31c96f07bb366d007bdddad72aa92f7ece90bb3e7e4bfde789bcd |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2b2a9dac416d2ee18a6fc663442d76871ecfab46c1830631b76d2f62e4dc7cbe |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-e4e183c8ebac1101459ff60df439aa8de997dd7cdae10fc1b82938cb4f80f5cb |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0a616dab | sha256-6ed2f211374199fbf1ed98dd1ab8aace8654aed751e7cbdbfa63ecf42d79d9ce |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0a616dab | sha256-a0190425ff05a184d7aec6cb64889ac0c85b71d200234f4074adc87f79d340ed |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-24dfdd98 | sha256-2eab5ad210a3e097ed3e96912986357483aaa1a734f7fae1d77b08fcc4a40363 |
