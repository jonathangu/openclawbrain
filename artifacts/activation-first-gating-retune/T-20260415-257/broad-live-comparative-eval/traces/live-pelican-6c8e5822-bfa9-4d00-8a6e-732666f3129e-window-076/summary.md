# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-076`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6f8a152004f1762e8eed1ecc9dffc7029171fd911a8be7a9ecd27602349fd8ea`
- fixture hash: `sha256-fef2416147c50461e059554c89ffc13514d9838c25717ac0bb496af10eda074b`
- score hash: `sha256-e143e1dba93875aaa62cf3e058a1fd51cb7a9d41edc933c5f184259cd8d7eec5`
- bundle hash: `sha256-96e3561e0e635a404d2b9bf9eab396f1a490e291abc0d56228e819c4aafb0181`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-510726e0c2b1103191bca21eb76122edbdd44953bed132c7c6febf953ec52703 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-dff3c7c539bb9c0652e8761189426289930c994de2b6714161d0063f64e39121 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a6f6950eeb33ce045859bdd53ee8104d21cb172c70838446e0c0e7528b9ed68a |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-70067f14395cb875a3ba0fb61818a18ca3133daaf7f139915593f76151815dd7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-434e1aff | sha256-e6c2fa04978138fc1dce2deb8e916580826b94975201f758d3d73ff6afd51c68 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-434e1aff | sha256-21440baf7f9332fb5b383798325f57edc1763ded6eb16773bcaefb9ddd80acd9 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-e2ff0c1c | sha256-f1f98372bb8c234c54647349481652a9c30474ee51056df52262d7a04c92fb62 |
