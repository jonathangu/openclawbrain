# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-fdd527bd79d12179b9a91214346f01f93616aaa30cfc7eab53977a331a071be6`
- fixture hash: `sha256-0aa39e409846ff84cb75f09fd340ba40a4ae31d0d07442053eabe16d211a0cbc`
- score hash: `sha256-15763e4bcfe784c5bed2326aeb6808632e11057343a0ff01d924aa043bbf4d2f`
- bundle hash: `sha256-0c4193310498810d30612ca6dfb70ff0a2bfffc3e09cb9f31f06cd71281b6168`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-349b3d6c28f24da121efce8d6fd84ec2564b6e3556e1440bc8512b8e1750cb4a |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-7115462e60e7a6720131ff9c0345ad5eaa0599c1566d7c308730f51f4b06c476 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-8b303bf58a0fbcadda0f3ff407f3700239e3303757dd6ac5099a6822bfe6875d |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-0501bc034a74280ddf00ff5205bc72aa8dfb8707e345f5628f1d7076290da982 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-83fdd5cd | sha256-51a8ad59d11fb082dbfa225b6b160c2b1a18128a53d717c482d8fecfbc71f02e |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-83fdd5cd | sha256-643b66720f846cbf305a73f14e18ea181a59fcd77fb107ff4f670cdfd31afcab |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-d665db98 | sha256-697d1037b76eccfa43ea56676a5421d36ed59b22301e4cf0d7f11de251475321 |
