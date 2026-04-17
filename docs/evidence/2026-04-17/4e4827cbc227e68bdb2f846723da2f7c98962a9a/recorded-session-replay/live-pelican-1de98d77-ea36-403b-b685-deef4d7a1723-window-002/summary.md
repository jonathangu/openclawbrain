# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b39ef4fc4945a82dff034380c9080960d0e6ed5fe56fe5b4657351529db21cd7`
- fixture hash: `sha256-a795947af952aa839da230500896d2e52bf78e338ce72dd740b6a925befadf59`
- score hash: `sha256-c3ae2278477a12e5d59597d5c5dd22c08d1efa24da05bc1eff5966204a0c3dbd`
- bundle hash: `sha256-6aaac3c480dc44abf2ddbf9ab401b3923a588778ffbe1732c4d3dc9f03a0a3be`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-64e7031bab11acf7ca7c6563e45ebf707e8feb9b8d59eced338f7e5e56bc854a |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-804e7dffa2c4fde579b81961ad6aa5a0c4ab1c12c4c3759178c2ccc09a1933c6 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-f11bb83cc1c6c7df482fe8fb6ac56c1e6ab123472cf4fa30cf4b4163a55560fe |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-1d25604dc40b5891f18410bab105bae78be69476f8052e2d016fbb34a587817b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-910a9218 | sha256-34b1ae7da70e7a36652ec221cf5172751c27b4513594ac8dbc4282cfcbf55409 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-910a9218 | sha256-6db9dfaf7cfd5562a750823513724b439ca36903459df19cd9c5057fedfa09af |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-6c1fea2d | sha256-fb25dd6f728bcf2b14f34eeb6b7ba749117d50ac42255fc5d212b1fa5d2b2239 |
