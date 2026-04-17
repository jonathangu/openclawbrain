# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-015`
- winner mode: `graph_prior_only`
- trace hash: `sha256-06c88cbd7b40857f6269dd03d5e04022f7a27c8c5e2a225bc79b2768cb90fdfd`
- fixture hash: `sha256-6d62bb5ab6456b9eec73e20f3d1a35ffc14e9452a4f4442f3b56ae134f63d27e`
- score hash: `sha256-5f2a495bba1bd0a8732a21063b829db4e95d97f13f5f5c36d5ec0330bce4d708`
- bundle hash: `sha256-c178e6fa21ebd93564fe43371f969529e889410341bb36cc6180d59ece23b355`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4cd163af8984e87c72885a17249c9a84973c54f74e5363d963d16ae86c9b4e43 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f6971386f6042a7daf40a051e68304c26b469b69259d9bf1f009184bceb48e4f |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5077582c0ef28fcc6ca9747084fdb4b6067e3aac74d583e9750c583592746db7 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-70013a4dfe4674e9ac853d97bf38bf44c965116a11276d40fb0447beef834c77 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-3434d724 | sha256-e2102d3ca819ccab4ff7f5defdad672687201599fbf40ae041d6aef35c3b2721 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-3434d724 | sha256-80c3c07516e4819027c2c2a1fc42d209573c3b350413490201a4ea9b04edc8ce |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-50a6dcc3 | sha256-f4e49ef00b2549b70b7291183e7f6d64b2d1dc5124d93d6a392536eb881a0c2d |
