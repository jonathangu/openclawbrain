# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-080`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ffb842d2d3e2bdef797f256817f8e1d78ce9bfb6aec6432fc2346aa3c074ed92`
- fixture hash: `sha256-198b6d169e431ec0de7f8f7799921b3db142fd3140222b6f3b7adb7cb8af186e`
- score hash: `sha256-81dcd1e5edf2f66f8f29ce35f7242ba082709b3b4c5ee3640dcf75471383d128`
- bundle hash: `sha256-702b5bcd3971aa48e36fc3c7f9d6be459be5a8d5591f67e1ca38e3427d629f40`

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
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1a3e358429af6a99da9bd2a17b2e259694dadef9e2e51be56f8572674c85b3eb |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-580199746efbacd14e2cee9e2c26d5b8eff1475d99995a5d9bfa1afe189c6ca7 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-cef16ae17f4b58402d75e63eaa66e33699aac6fc0a5a958c9cb6bac90e09e5bd |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-44d333c1 | sha256-f5f592388a84d6a7be1b736f5181a8e92d6eb2b35425ee6c180571f6cdebf0d4 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-44d333c1 | sha256-2e0f88160d37585af170a1fe16b2a2cb7d5e769b7a1d91cd8e6b3eaae7d51a81 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-725e184e | sha256-87eca6eb00a318627efecfd78c833d6b7576d4cd269b0b8caea640052191312f |
