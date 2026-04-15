# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-fdd527bd79d12179b9a91214346f01f93616aaa30cfc7eab53977a331a071be6`
- fixture hash: `sha256-0aa39e409846ff84cb75f09fd340ba40a4ae31d0d07442053eabe16d211a0cbc`
- score hash: `sha256-62c424769799d785a378ac643e5ea905db8f338970650e73f9d81bae7f0af157`
- bundle hash: `sha256-54ce422762e9b18e30da26bfe77fbf6cccd47be1b031abeda6ca9f5817993436`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-349b3d6c28f24da121efce8d6fd84ec2564b6e3556e1440bc8512b8e1750cb4a |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b6f6f0debf973f4fb0315b8604459b3497a06f1e89dd9c0fbf9b72d20c3f698c |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-68aac7fd994c806acee84d50d1ada9d10f7e5721af6de3675c3dc5d55ded75a2 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-3c08182c2610bc1e4735368e06cf7ed9c55fcdd1901c95520c7f154b5816fdfe |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-3ec24aee | sha256-878fa2a567d88c70a3820ea5a78cbcd639754c26f7a43139af82323ad7fc7f83 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-3ec24aee | sha256-7fe4a9f8de1b904562e9a94fe1a4d501434657f86ae972bdd3f14055ccac1b75 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-3ec24aee | sha256-878fa2a567d88c70a3820ea5a78cbcd639754c26f7a43139af82323ad7fc7f83 |
