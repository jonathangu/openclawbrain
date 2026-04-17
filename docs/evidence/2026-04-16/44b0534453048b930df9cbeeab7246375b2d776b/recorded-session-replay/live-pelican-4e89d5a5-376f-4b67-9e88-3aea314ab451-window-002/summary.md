# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-4e89d5a5-376f-4b67-9e88-3aea314ab451-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-11f89d0770e58c74a32e0ac08329b409440ac220cb647ec446567aadc15cbdd6`
- fixture hash: `sha256-7793c2d77fac055a1c7c47c9d026a76a01511a45ccb17bbe5db49943de3d0ea4`
- score hash: `sha256-09e5094cc7862c15b4bb83b5fa20188083c36708d3b99ddd2882ed4a420d4db8`
- bundle hash: `sha256-ea7ffccee8a7f9a0ff526c7b013f0bcacd1423d3ef68161965a9622400f8c812`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3297dca5d83084645cc80493377a366cf545c5142415159b972c4f8430720ab7 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-0bec16b7c1ae56293d17634280a625c8f40fa0a1b934d0c7ca5819130308a364 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-afe09f1601a0bc20e6dee3cb6547b3db4f9fcf665b47d0af16748f74f25baa2a |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-c20b160c958fd72764a6a445c68840720eb99930da46b68e49d00d14e5770b18 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-542596c2 | sha256-c4786f43f791ea526c2fa0c7abd524b8f42804a1f95b2ed7e7b571da66eeebcc |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-542596c2 | sha256-8da005312baaa89a59301eeff80926639f2dd3deb0c0484830aceea4df58789f |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-af967cdb | sha256-4dae5c3ee7b2f35eb8555e869fc1dafc88d9323f11f50af29078f08486211604 |
