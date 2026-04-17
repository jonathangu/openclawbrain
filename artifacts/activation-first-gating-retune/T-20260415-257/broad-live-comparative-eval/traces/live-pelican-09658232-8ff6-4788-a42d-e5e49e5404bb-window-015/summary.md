# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-015`
- winner mode: `vector_only`
- trace hash: `sha256-6e0ff46b57f7c50af20d23a4d8a3f648535a36cc4021c3842ecad10617366b5b`
- fixture hash: `sha256-d2c3dec4ca105c441633ffddbfc56cdd05e4790ddeac1ae7cb9c9e93a7fe734a`
- score hash: `sha256-1efdf2a64fbfded90a4d6dc9585d02107588a7ebcc87fc5cb19a6cbf9091eb18`
- bundle hash: `sha256-ac80a07149a67cd0d1452e22cc6acb8a9e689d32d46da6c58cb1742d7530ed48`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | vector_only | 100 |
| 2 | graph_prior_only | 40 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 1/4
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-48620826d5383928480bcd6e76b64505c9f9a20a21654ee2da4ad581ffb660b0 |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-7b4fdb2dff9bd6bd9c379e7e78f4223f192d478d07697edc023c2fb659c7117f |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-57edc80c0f79be2ed7bdbb18d36bbed2365ba05d127da12805916ae28b1ba247 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-ac5eab96c8e65c43d1a709113c3bae86010e64f7d0a6ab681b9bde2a258beee5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-fc827bbe | sha256-f3607665091f2579e082841d5711b94a6e99581a16335094c4fa8740ffd0bd2d |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-fc827bbe | sha256-0ad5fcde6725809645645070cb80f28b5a9e36dd1680b0eeb0a33013f278b38e |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-5a990a1d | sha256-d20e58c30ab082dfb214b69ee07db16ea3f649285162c577baa0657256b8f413 |
