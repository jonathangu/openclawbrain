# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-010`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bd69cb2bd54df203880c8fab0fa4c855341f06ecfb9d6ec642144558419aa71a`
- fixture hash: `sha256-f29630fbd2f41b8d395fae06865eb7778e00433b1298788381332e0703a42702`
- score hash: `sha256-9a78ab20bf6686d4976c94b46e6767a137b886ea5a7a4647d7febd4db070b295`
- bundle hash: `sha256-ad28ead0cf5a6218c70ea3db321c0b488ffffdada72ca8448a065a2be51d3bd9`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-80e65d944c503b7cf482a4ac157c70bd9810fcdc3cd3dc77c36042f87f3356ea |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5230e1fbee1241754a20cb4e024a165713dc5f7de610bd2c82a8e67946c9f046 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8e2f8ab9f7ededc3a416b250ab0863285319e47f978e93902696328bf506e0c2 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-a21ee89f1437c75da3c4abc644146dc0bdf041335ec06d260f5bf54b1e461933 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-c6da3b8a | sha256-431d806fcaf06d26d2358becb5afc4becd212e3b933cf6002b810a3858d10fb0 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-c6da3b8a | sha256-86653f6098d4750ca7b6047967b9c051443b2a9ba2dd099761ab8745a5bed021 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-01e92d4d | sha256-b8e1df9702dac3830c2794301e0cb5af737266bbef00f3a9681f03ebc58452f1 |
