# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-035`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8ae788de26ca53295ab286d92504e01df10263a019ee2527af469aa665e03d13`
- fixture hash: `sha256-40c450ed66f286026623777e121b2767ec1f98a9a30d5cbc431b359ded23bd1a`
- score hash: `sha256-626a2cb0dca916badad7485074d74e2dbf7714eaff3e6f1331d8c049ea2624aa`
- bundle hash: `sha256-21b2562dd57184e524c96cfd41e9d348f97cb53948472f671c062732e8c94ab0`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/4
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 1 | 0 | 1 |
| learned_route | 1 | 1 | 1 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1623aa98a961961a182098cbb09dfbf96da5584b9efee0863f57cb38d7ebe41e |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-91a1479a80f530a30994aa6fc6b6b4979aa4b5eda5dc561e60f00b8fb1bebea0 |
| graph_prior_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-9343bacb9b334c5970cb3f4a6190f7ccef79ed66263520f7be9e9f329a7e2699 |
| learned_route | 1 | 1 | 1/1 | 1 | 0 | 1 | 0 | 2 | sha256-1d6c6a9572f85aa0eb9c97e2591a7c930d331f5ca085683a2f2fb36949e747b2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-6bf1a42a | sha256-15d443b00f54adf81dc01f99920b020568df5c52120ce18921dfa5ebb850d33d |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | no | no | pack-6bf1a42a | sha256-d9a928170e2c442e434e1202470adb4ba7133ff4e392ed0dc31fd8025fe9a0d0 |
| learned_route | turn-1 | 100 | yes | 1/1 | yes | no | pack-8f8e7b5f | sha256-45ec1ac45afb03209136090b781ddba9c02da7dde9de7bb6380ef38b725a2e71 |
