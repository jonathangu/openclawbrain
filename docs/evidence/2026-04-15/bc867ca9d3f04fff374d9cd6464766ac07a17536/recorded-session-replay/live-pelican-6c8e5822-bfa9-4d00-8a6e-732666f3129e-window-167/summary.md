# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-167`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9808f8fc2b34b9bdd2037973a4af69235ae13bd43fa03c06a9cd2c930faaaa29`
- fixture hash: `sha256-c59bcee0d7e5004e8699b7491ca609cee1baf21baa2824b5dbd8c966b365083b`
- score hash: `sha256-59a0400cf09b96bde04d03a590c8ad3789ce5120165cb3275191f1a7f74de5d6`
- bundle hash: `sha256-c444ec23b3e1ca9216b05f7f901d09e944c5951914a0c92cd0ef6644f816c95b`

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
| learned_route | 1 | 1 | 0.333333 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-68b72eac2031bede9aa8770eaa1f000f5f4b3a15976311e630a954289393b0bd |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-b85862905a635043303111af32b93f126ade1911f16433d28f113237a8560d34 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-c627af784bea457788030be9129f7d6e6cad1fab5d16c0653b4d9bed04e8143b |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-6e165bbf5c5962f04ec24970f666a0a433d58ed7aee93288e7d6c0b3c0cb22f9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-04483d8b | sha256-5cb3ef1a1c604c63daaec31d0bf3bda823c0f6b7ff4121c008ff5d6a0da625d4 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-04483d8b | sha256-5ad90e6bc2e87964bd925b778511f9db9fa5b2d0e5ebd0cb2ee7e62bb1d42a9c |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-04483d8b | sha256-5cb3ef1a1c604c63daaec31d0bf3bda823c0f6b7ff4121c008ff5d6a0da625d4 |
