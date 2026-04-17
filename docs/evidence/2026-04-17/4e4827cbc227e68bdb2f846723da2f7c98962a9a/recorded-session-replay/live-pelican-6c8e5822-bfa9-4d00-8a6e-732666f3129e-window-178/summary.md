# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-178`
- winner mode: `graph_prior_only`
- trace hash: `sha256-dc837ac64ce4a5cb1d121e2bea7830254f5b1cd1faf9dd8be0505cf94fe18342`
- fixture hash: `sha256-555eb18092c7a3b48bf36359187522f84e12b063bd73ce65d859cb8f468c2af9`
- score hash: `sha256-d488e19a428f4685b6712e3d0cc1a5cde24744fcaf09e570f722ae29f15143cf`
- bundle hash: `sha256-73912fe10600b7fb08cc21aa7f3ec92a0a78518bcfcac3b5511b0033c54cd889`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-94018db213a88670c23984311d9a8431beabced6aba3b25434ee10a70b79887e |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-39ffeec2a2982549c3728ad425863b1f269497ee9a956b4a199a2300af960a44 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-41bf38c7a9410e9330e8fd5fe1a4991f20f87ab4ccd5333ddc71614dfce84d18 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-1ee5c3468fe6cffa33dd4b1f8f1a105fac5b990cb5f433e99344376ae9fcf5aa |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-52bfd113 | sha256-6ea3a8447d392c09e670e49bd126ffe6784ceb4a0fd592ec77086103db159c87 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-52bfd113 | sha256-ae62bfa3073f1d22399bbb4602769a304033b2dfb303e75d14bdb40ce5bbb2ee |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-ef7256b8 | sha256-fcaf2adf867a490bc83956a98287e4a87b6b9bd21ed82007989f79a8a722b8bd |
