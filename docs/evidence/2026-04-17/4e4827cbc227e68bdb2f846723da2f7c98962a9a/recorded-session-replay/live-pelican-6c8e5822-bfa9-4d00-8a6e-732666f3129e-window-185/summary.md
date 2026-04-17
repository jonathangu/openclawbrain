# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-185`
- winner mode: `graph_prior_only`
- trace hash: `sha256-24e1e9ea471d19d207e35c598683b69d84119849186e1c11e6ddd97932c4aba2`
- fixture hash: `sha256-11642bd40eb6fe8c9d53921bdb1bbcbbdf6e5f35f00a6469f30893bcfb466a96`
- score hash: `sha256-ae468d14cb8ebd9dbbe1959f9f094ce69402ffab3baa9910093b7365d557974b`
- bundle hash: `sha256-96bc5feca8b2a08847b4606f6032988d5e8375c0eb78049e6327617f5378d5d8`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fe10a677dfb68dc498fb14e838ed3e08e036ad9f9df81513ada323fcaad39838 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8ee641c4782931f7a1d6f67420e96cfd74e285ced9022168a0b2900f986cf1a9 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9ecdf3c36e7c2fac8a2b4e479adb78d1766e0c6358faaf2af89f0f69164448ab |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-44fa3cb4fe5408bb920b48b3f4727a9565a7fd4b0a2e327ddc574b6b8fe3c4f2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d6332855 | sha256-45746c6b1a4de504b8083f1a795d32754992f71a3a01271ce5499cb87c0cece4 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d6332855 | sha256-f9eae73fee3312d3848f98a6c011b0739cdcad65059b249caba4018fcc4cf3ab |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-ff944218 | sha256-4b0c7510ea4426d972e3fec61fd1ac1ce13bbc322d9bcbe0bd0baee8c3d6afbf |
