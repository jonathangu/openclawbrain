# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ae66dcd407454d64cc41d28f61a0e11b77513e2814da48c47efe5c8c6e3c8baa`
- fixture hash: `sha256-8798537b3abe1b5c15bce4787c7758c4cd08e15c5c204adce3a372ff88067693`
- score hash: `sha256-45d8d91ba290d6daad4f49fb62a6a0dbfc94f3295f8a20399b9e547228213632`
- bundle hash: `sha256-39ce788f9c96170639397594a1205c91c89774abd9258b6f56da7e5d0d22114b`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| vector_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-e89c44412665b2db521df92d2f3da0746226286a7982e82777b52b3899520897 |
| graph_prior_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-22a8a432e381810ba16eaf83e634859f44f2d33710ce95478bb93c03eab50695 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-6dd554780eb85db56e9b469bcb4ec262ef5ecf6d0008973d49584a64cb7767e1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-22e27230 | sha256-a2e0f3816f3711248ced914cc32d668881712e8036894ac019fb460e109fb3a7 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-22e27230 | sha256-5bf47f3ae3b0882ae13a0ffb84a8df29d2d556d7476967bb0a6ed964cfecc46d |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-22e27230 | sha256-be94168136f7ec1c070f86e9cf2273a5745e62ec6aedb933653cc2f7dbdd42e8 |
