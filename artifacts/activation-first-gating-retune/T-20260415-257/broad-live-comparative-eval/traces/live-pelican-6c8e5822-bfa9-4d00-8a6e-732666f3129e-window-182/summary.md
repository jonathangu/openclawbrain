# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-182`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6cd785628fb3c34642dd7b4a701799a6e96acb06e347a7bf1d01cd4950a8de4c`
- fixture hash: `sha256-8169ace4aebbc5a4a546b5c0d2bdc7c5a395f1f1630a066be79c7f63594673d2`
- score hash: `sha256-dafcb59f2fe8d251f00905287485501a2eb99e704334809cf20bd4550de1dff6`
- bundle hash: `sha256-70f179191f365e1da08021a8463a850226333861c4c3666449ff4812685a45a7`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 80 |
| 2 | learned_route | 80 |
| 3 | vector_only | 80 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 6/12
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.666667 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.666667 | 0 | 1 |
| learned_route | 1 | 1 | 0.666667 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bf6fbe7613b07ee3e659c5ab0ce2fe9e83640dc0dbe17b255f0c268784354a36 |
| vector_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-d3f65a92c7bb726d2d0c2ce113d539480addb3c84cfb3644c93307af7ad139ad |
| graph_prior_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-d61d891857d1423b65fdd95fc119416917497229c16ade3d42a408fac870239e |
| learned_route | 1 | 1 | 2/3 | 1 | 0 | 1 | 0 | 2 | sha256-4174f7c97c604d0fd7f0a2e92e8646a19bcc25d7a72181240c3bcba905752651 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | no | no | pack-1c7de9ce | sha256-ea20f08f8119b210883457ba38ef0035bd337e1b90d5197062d42575f673c34e |
| graph_prior_only | turn-1 | 80 | yes | 2/3 | no | no | pack-1c7de9ce | sha256-371dc4682263cbbcdb272cbac46ea45c8708320e8a1e4c851202412b5d673a74 |
| learned_route | turn-1 | 80 | yes | 2/3 | yes | no | pack-d3060395 | sha256-49264bf299ba68756a84341367c3a8c0415517ec5ef23f12e71da7c4193b90c9 |
