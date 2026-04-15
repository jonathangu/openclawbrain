# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-029`
- winner mode: `graph_prior_only`
- trace hash: `sha256-766d9b6ce430d9d07fe2ff3297e9849fe05332d7539d3d62db1cee2a9f89081d`
- fixture hash: `sha256-21e8a90c2dad8ab78ca636bf0f382e5b550e2af76a7681917f1773769c731648`
- score hash: `sha256-aa011b0f7f3d98641ca4574d5951a01a690b0c51a8f215f81293a1886cb531e1`
- bundle hash: `sha256-9f53a4872aa4cb2b142409de2001d551a83f71d522737ce5b302e950e4063350`

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
- phrase hits: 0/4
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d8021a8424a98c9c0ae913d23bd911fe66b4179fa226e5ae4873cee34e53cd89 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1d68e70acf6a414218753e122f91496fdc4aebd9b46ac953cf121ff4892fb4e5 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fb1368aa51bd74b00b2892fd6c26617a10cda100c33048b164f40fb0d4e3057f |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-02b994a3c88d7573f1ad305d47651a205de0689b2616e53e0ed5106a5d5808cf |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-554f3072 | sha256-c92b9c56ba54f3e0f5fe659542afe8d1eded0e1eb494c9ecd48b4d85f40f3277 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-554f3072 | sha256-993cc715dea86b828989bd206c07844593a287c95ce077d9e7703df93e030874 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-554f3072 | sha256-c92b9c56ba54f3e0f5fe659542afe8d1eded0e1eb494c9ecd48b4d85f40f3277 |
