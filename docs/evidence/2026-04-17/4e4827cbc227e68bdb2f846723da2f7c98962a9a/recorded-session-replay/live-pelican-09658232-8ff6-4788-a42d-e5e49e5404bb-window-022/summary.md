# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-022`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f7b973c64c8eac5a0b6deba25fbae9f31be4599e3d19192c7c9dd0b18e718f1e`
- fixture hash: `sha256-b932d5e627b7081f980ab111b252e205aa7e0185bfcd774e6388fb9e948098c1`
- score hash: `sha256-70dff892d78dff0738e63909e99f2fdc9bed11c1df147498cc4258b2c354b832`
- bundle hash: `sha256-5105707b355055ab812fe4e7df6b73571b4ad6965baebc72b10a63a3e4d2b023`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-fd3c28bfccf2817f3d01d14dc16c97875abfde806e8cfbeff2d04b6e2a397e7b |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-0050a73c1c8b7764218879a4f1dbea356f3545e1be7cd3c21d95e72fbba5bf8c |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-68fae15aa5a091b761cdbdc7a065c2575b4bd0d3a489862d97918b25a75e2de3 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-fa4c76bbaada42bd476e1d6512adac89ee5654f0734f37528bfd375c23d0c241 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-825a0e0a | sha256-bdd1f47cf128ab976ac56bd7e92e6b7661da48ddfeb68b52bf7d45a48590f94d |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-825a0e0a | sha256-bdd1f47cf128ab976ac56bd7e92e6b7661da48ddfeb68b52bf7d45a48590f94d |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-584ef1fd | sha256-0b50520731cba042471fddb61770d4a4a2fbe96b67e888f62a1786f7407e01ac |
