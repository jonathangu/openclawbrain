# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-035`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8ae788de26ca53295ab286d92504e01df10263a019ee2527af469aa665e03d13`
- fixture hash: `sha256-40c450ed66f286026623777e121b2767ec1f98a9a30d5cbc431b359ded23bd1a`
- score hash: `sha256-404eba733f59e2922bac1680beb3b1bc3d0cf694e013d208930e83a230dd0595`
- bundle hash: `sha256-5e3f20757f45cdbf0b5271b2975e8aea0d70e023a7749562f31bed0b1343a982`

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
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-8eb7b5e7638c2f911ff6560a5e99bd0de2ba678c6111fe585c9be9edb77fa4e3 |
| graph_prior_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-dd85e745ab6b3a3d4275e76f52490c40c77d746475fb78bc9707ac6778aeeb45 |
| learned_route | 1 | 1 | 1/1 | 1 | 0 | 1 | 0 | 2 | sha256-050f5d42f7448cf1502ca71ca662409891424ff23a587dbb4a266bf089be2476 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-5ea85fbf | sha256-f76524a90dc00d0d9330ba2fbfe369509a14e0e1b0747810b34fdcde694ca335 |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | no | no | pack-5ea85fbf | sha256-36c1d18c9b0063957c70ef28e27ccda808b2b9635a010fde216101316dbc2fff |
| learned_route | turn-1 | 100 | yes | 1/1 | yes | no | pack-824536f4 | sha256-58f63b8f07a3209a419ef63d1c5800003662be876d82631717af5780ab78090f |
