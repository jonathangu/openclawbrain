# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-023`
- winner mode: `graph_prior_only`
- trace hash: `sha256-858b0f43ef470e5eca2afe1da13983b8601f538afa969d1fec1c6c995e06b43b`
- fixture hash: `sha256-5c10bbce6c643d206da3406e24da637074a598036c8915b8615377d8cba78cd2`
- score hash: `sha256-3a7e8457032f7105362c57a5d01cca76692f2a5f5acad87135d602ccab07dc8c`
- bundle hash: `sha256-d208d58e77b0631cfc0e5beeee6745c3d3dad4b7a12b326949fcfb009768c9bd`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-17f51076699069dddd3d95709feed654148e93f103c223926ea5c4a4e2da537b |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-dacea337ec2a883d53aee114ee3905af5987580a132d4f38313dd07dbec3e388 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3f13aa7eb52d7034e864c9706e386ae7bfd041378995ad0e862284305c1d29fd |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-c4ad3fec45943592184c6bd13e7e7474c24f147fab4d34beecd9b47d829d6f76 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e9720dcb | sha256-21ab4911a670bdeb08bdac6ac7672cda09b280e08cdf6740cbc5c24f6dc7e4d1 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e9720dcb | sha256-8b95d4046f5be99ed6401b3ab7cbe1d9fd6f80ec40206c7f9d774faeac3586f9 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-e9720dcb | sha256-21ab4911a670bdeb08bdac6ac7672cda09b280e08cdf6740cbc5c24f6dc7e4d1 |
