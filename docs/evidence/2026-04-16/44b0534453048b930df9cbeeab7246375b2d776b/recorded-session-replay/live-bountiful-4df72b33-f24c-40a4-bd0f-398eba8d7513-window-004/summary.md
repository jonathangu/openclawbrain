# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4b9d58238866be4c30cb67001ed41c476bb074abc457ce91f27bdf2a95087dda`
- fixture hash: `sha256-93a191f41c9134f7fb1b39f4120c598d79722f0fdf720a1c60726eeea45f85a7`
- score hash: `sha256-60b4d0a399e5c77b86aa28ff4484e488191b78944f0279b36542c5b7156b8fb6`
- bundle hash: `sha256-7231df51a4d54207690da499595c9a2a5259531ae899dc0cbb63acb6d267115b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-53ac170bbdfe31610a82a7fea6a20f739ad327e9856e23aa713b46f86601ea52 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-e93fd673f513a11b3ede9219e707a67630163b5c831e6a01ed43c3af3fae9ac3 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-ea77df7b02f17342d6793b6af29f617dd8cf7ae39cf8713f74ae4de254e2ff44 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-601098356206707662aa23132ca55ba2e4ebc44e8124186ab1260fe194d19fb2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-5eab2c6d | sha256-018224f8bec4f7c95b96db7d7c50286b9c44c54f5194557389458066979890d9 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-5eab2c6d | sha256-498ded773454232a4b8649923d7c53d1c8728398061cb1db0d95118de8ca6661 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-3d22fb4a | sha256-7986cf7992c64a42d8ec7e9fd8ad0a263a9363250a57b33a670225d5aed533ee |
