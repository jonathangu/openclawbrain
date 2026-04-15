# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-017`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3ae37b035b14582a9db80eca92cf9ea284e1a083da7e40c15c27766593c501ab`
- fixture hash: `sha256-c1a05a74d2fece7febaade02118c0528463204c7c70c4fc0e050990958f60a91`
- score hash: `sha256-65351576be6dc04d08e0f86d3e881d768a2e96e883f2c68134c97b3768fea5b5`
- bundle hash: `sha256-fd8eb6715dcccb886880539a87c659dece086bec3dc5ee5868bf3619ca3b0072`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5bfccd81fb07da2148e74d03332b298ae8343e32f9c89c9de3c815764af2fb42 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4b92a33e797403731a6b173a5f98a268debbddd3eb9d61cab91ee30ab55b7f19 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-292b42f62fc3404ad76a1ed09f94356fc644ee68e4af1adc791758d3f0ba2ea1 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-b1b4ebe5e114a413a56531b284276866abf788bd31b34d724eca8a720de93d35 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b2dc00e4 | sha256-607c3396cacee0102e55ca26b4918a43654ecd755ceb4d98257bca1dc0eec692 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b2dc00e4 | sha256-a6a23b05cfb2e74aa38ff647b5a7a877d1db2b6e4f00c9ec6e4242af00b3aa32 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-b2dc00e4 | sha256-607c3396cacee0102e55ca26b4918a43654ecd755ceb4d98257bca1dc0eec692 |
