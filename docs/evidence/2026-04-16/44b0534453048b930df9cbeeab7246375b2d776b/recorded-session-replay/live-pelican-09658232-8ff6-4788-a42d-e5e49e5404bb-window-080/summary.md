# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-080`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ffb842d2d3e2bdef797f256817f8e1d78ce9bfb6aec6432fc2346aa3c074ed92`
- fixture hash: `sha256-198b6d169e431ec0de7f8f7799921b3db142fd3140222b6f3b7adb7cb8af186e`
- score hash: `sha256-92215924756a1efb65a179282241bb39f8da2ef6fa3e433a5567547240c74cf7`
- bundle hash: `sha256-183af1919cb7285e0ecfe15cbae9565242aab784638fa9fb07ad3e27e21141e9`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-236480801c2c951ce502606c7421d96d831d23f14e852b882e65d08e48147fac |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0cbcd74304f3a39f25d3c7eb690956dd406d8b3fb304a92cd201db4e52167e74 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-04d35e2b07c578c27f220eabc87504240fe39a5d09b55edb70d0068fd5312b63 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-fa75846e31b4dd6af9798716c5ca30026234a63534a6faedfa6d89bc61f64ebc |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-521c782c | sha256-212e9acbe00758acb048e144a1b9874633c234131fc8189369667ee6e53ef517 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-521c782c | sha256-ceff308487895f7112ef78b1806b5e0dcc25f8ecd4ba497c374a14406bf81ef8 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-7fa75cb9 | sha256-1ca7a7aaea38eb4dca0ff342a571f1e79364da5952c10e8b6527de798e8b6346 |
