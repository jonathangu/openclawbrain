# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-185`
- winner mode: `graph_prior_only`
- trace hash: `sha256-24e1e9ea471d19d207e35c598683b69d84119849186e1c11e6ddd97932c4aba2`
- fixture hash: `sha256-11642bd40eb6fe8c9d53921bdb1bbcbbdf6e5f35f00a6469f30893bcfb466a96`
- score hash: `sha256-61a84ccebbcc5d0800d35eb08e397480bec911567156be79ffc83b5a6dd1516e`
- bundle hash: `sha256-2501499b1a2d5f9916f0477859e9e4fc27923ea4d6106e0ea55b8172779ccde1`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fe10a677dfb68dc498fb14e838ed3e08e036ad9f9df81513ada323fcaad39838 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-638996ad6cbe5560312559cf142db04971a4509a9630d216377c1808bd973b94 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-901e449156b43a76f801d13880601e37b34bc728c6a0dd152ebe99023fa361b9 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-dcf4fbe5bf8a6be94eab559372fa197ca6c3df2a4fd05ca6e62706d6f871b446 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0847e300 | sha256-c95fc9618f4890d9842c44f9d888c2ba0b7c71eafe0d64365432546dde517b35 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0847e300 | sha256-d0f031f27987706c51fafc5aed00149acae669a07cc8e3e6437f3bf2227ea578 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-31a8fcc3 | sha256-31521c970cc759cdae4af974929ccbcec0d5198f93f03d47c1155bf0340d7c8e |
