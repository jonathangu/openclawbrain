# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ae66dcd407454d64cc41d28f61a0e11b77513e2814da48c47efe5c8c6e3c8baa`
- fixture hash: `sha256-8798537b3abe1b5c15bce4787c7758c4cd08e15c5c204adce3a372ff88067693`
- score hash: `sha256-82a8ac6065622fb685da85d1ada5f57d883828f23f502c4dfc2e7b6ac5922053`
- bundle hash: `sha256-92c89d874733a9bb04ba0a0566f3105e460b6dee8577447acf3e3087e3ba99e5`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-2631460258b8349bf5851bc29e43a192f583fb77925024687b67768874305033 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-e439b55b270201360a7830522e54e3e4845fdac1acac22265275cc801cf5f86d |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-1b8185bed7ea39f75af34e69502b6b3027a4ab82199457db4f327a41ead5756c |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-e4fb0d9dcc2c51ceac2252fd4aaf7e09c7b6865ae46f35e2820064d666578c52 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-e15a7cdd | sha256-092669e86e4be09ad1e2f82f28d5b16f17bd9714d641b6a1799a2a2ae6b57117 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-e15a7cdd | sha256-7d3a4cbf8c92be83ac868afecfd68f70523f311171fae1c94761f59dc3dc8f18 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-a7c3bb9c | sha256-df9ad43067e3ab0ffe47edcd28487fefdfabc5bd94e379e3df9c6e24f76f27a0 |
