# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-080`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ffb842d2d3e2bdef797f256817f8e1d78ce9bfb6aec6432fc2346aa3c074ed92`
- fixture hash: `sha256-198b6d169e431ec0de7f8f7799921b3db142fd3140222b6f3b7adb7cb8af186e`
- score hash: `sha256-5d398cb43e7f94bd165b0d9917c4ad7ca050243badb0da60be893822a579a499`
- bundle hash: `sha256-0d33827f8132afb4ac552124274bd9f94c98276bd38765e03d3543583d739a95`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-236480801c2c951ce502606c7421d96d831d23f14e852b882e65d08e48147fac |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-27519deda398b6b6132f8c25c0bce8714cdaa2688b98960b60a6c782d8b69e43 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e0f08d6d042d31d3e178218e6af7eead4a86b2bb49febe0938e6a3ca0b671438 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-3f184be335a4386d00f2be0731eae9ca4ef75a8b7973bbb9f382d3d3793ece93 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1bde35ea | sha256-e2aa19da16355c5e5a9c773ce92b7ad32807ca483d4bd7b5378ac7fa3fb5a2b8 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1bde35ea | sha256-cd4168586749137e6f6b8e81ed5bf506ac452b0c8261cef2d70254953760db26 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-49691a77 | sha256-433b30e87e984247c98e3b98f71130636dd6ef4d016d80c0e17a86a5941ee3e7 |
