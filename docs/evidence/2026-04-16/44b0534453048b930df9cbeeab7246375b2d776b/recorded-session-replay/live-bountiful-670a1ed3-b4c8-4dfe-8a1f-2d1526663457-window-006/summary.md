# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-500bb42a51fe35739e28b1f6be3d9fe7ff92c6a8eeb2f053f3018ae2eba88584`
- fixture hash: `sha256-f69dca5c27c722f582ac3debb2e25adae4c35c5bd6a4749aa476e37eee07c7bc`
- score hash: `sha256-3ade80b51c2397b70012698e9741ea7779eaae3ad22a5c7bfc06526e0ca2e681`
- bundle hash: `sha256-25f984fd5f059134a78cb928e3e7a5507b68adec333d52c75e7f3198256daed8`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1a93c94aa4cb26ac67e3ba4bdee5fc22bb0276c3da7ff11089c43e42405c272c |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-6a59f65dd7e362640a0ea16d1c6bbfccab02b793ae5f189dee2ca8a29e1d962f |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-ffcaee83db09b1676b745cfbb96ee5682ab6ef8b678ac24a485f9c3289d834f7 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-85deeed11bae071b4b5805470e5ce849fa278d2008d224ae37785e6f9a09f84f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-9741d42e | sha256-6540a6ffc7ede4175ce62400200334e19235ae26fe6107932285f090f6f7f8e7 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-9741d42e | sha256-c86a6d8bf9e10f5f867af59e47d30cbe346c7c9b79fe77b8f4396f9ef317e49a |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-d963b427 | sha256-0a3c2df93e68863bacec6929a8a65206142ccd56818f51d1c60d4e3ea5032648 |
