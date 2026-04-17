# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ae66dcd407454d64cc41d28f61a0e11b77513e2814da48c47efe5c8c6e3c8baa`
- fixture hash: `sha256-8798537b3abe1b5c15bce4787c7758c4cd08e15c5c204adce3a372ff88067693`
- score hash: `sha256-1283ec930261678cfae29db6aa8d9c9e2104fffe6d2371d4cd9f3bc7bc84bd82`
- bundle hash: `sha256-541dd0100d057a70280fcd157a632c9529b44fb792ba9f907fddf8f028248aa1`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-2631460258b8349bf5851bc29e43a192f583fb77925024687b67768874305033 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-15c28b7a2bb06ac1abd08b7a27703c54b8eccefaa809c1af38f5a9b6fc8ba1eb |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-e6779a95ed829c267601d7c6833ed362a457a06cb5a9fb603c8fe77cf9023630 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-7807815e3d9d2600bf3f716995db82f51d5db1f8f0ccb7c61f44a0ddede8cca1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-4880db3b | sha256-d99bdfa3f53e2fa145760aef75a148f562063bf5c2759f16cc3c263858ca5758 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-4880db3b | sha256-05ef78cead5c78ace5ae53ff72e42e03ee4259767a9e3a99df3d5e1292cf15d0 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-0eea19fa | sha256-0261ea2ef177f5a2b7d3a59ddb888bb83725b641d77c1343c646d0699e28e3a7 |
