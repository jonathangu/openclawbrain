# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ae66dcd407454d64cc41d28f61a0e11b77513e2814da48c47efe5c8c6e3c8baa`
- fixture hash: `sha256-8798537b3abe1b5c15bce4787c7758c4cd08e15c5c204adce3a372ff88067693`
- score hash: `sha256-0fb1a3aa98696e21885055f0a121378d17f4c51a0c6841e74639094bc7c72125`
- bundle hash: `sha256-f7248c61aa2925ccefb82252d4d6d6565e220b5f77e2ea68332c44c27139a008`

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
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-c39d44fd200449fababddfe19b65ea66ed9caaf1f173c2ee4882b08b529b3ddd |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-2ec4bc75d971065b05795b228b125f75e81bbd4710a6f77b874086aa71df3b9e |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-3a46b48a864f2c6edb54979e36bc52fe17fa86b8586520dae21a58d5f085a473 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-11b4137b | sha256-642fac2be0d28fe31f89230f618e8f324032ffbb56cb07db33147de2d5ceb541 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-11b4137b | sha256-ac9e500b1107acb533236728c0044384122aebe5103729752f1448311149c494 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-11b4137b | sha256-642fac2be0d28fe31f89230f618e8f324032ffbb56cb07db33147de2d5ceb541 |
