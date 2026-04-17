# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-035`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8ae788de26ca53295ab286d92504e01df10263a019ee2527af469aa665e03d13`
- fixture hash: `sha256-40c450ed66f286026623777e121b2767ec1f98a9a30d5cbc431b359ded23bd1a`
- score hash: `sha256-5ff07f0578a71eaea02f7ce6c0948b324842ab7447c0c552768e77a1aad20aea`
- bundle hash: `sha256-59aaf857d1cacd183ff2618f1cf2d47ba12112887ab02bdcbdbf50188f83b7c7`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | vector_only | 100 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/4
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 1 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1623aa98a961961a182098cbb09dfbf96da5584b9efee0863f57cb38d7ebe41e |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-f0d618e750c299fd9af372e76c425d1ff7f44127199fb676cd8cbec0891ce36f |
| graph_prior_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-4b997aad967ff0113b3ee0ee508fee0b81aad884a9cec8ffa4cef9c3f9e1593d |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-a782b8bf5f60cdfd67ce7fcbc16fbfd26ec1a12afcf14bbe9fee11a5c0e008e1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-9510c433 | sha256-a303aa810a7b458e8a6dd1220e6318c60470f9e7b5d7c7f18cb6be40bdc8017c |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | no | no | pack-9510c433 | sha256-63af33cf5e0e34ad0023da7d305758b718ff334f7afbfe65e0a7462e932d487c |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-b8ad9b68 | sha256-422baf50dda00150c3f82620e9878f73244e210d50bba4628fc8c5fb1b3fb44f |
