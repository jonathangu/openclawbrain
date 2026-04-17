# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ae66dcd407454d64cc41d28f61a0e11b77513e2814da48c47efe5c8c6e3c8baa`
- fixture hash: `sha256-8798537b3abe1b5c15bce4787c7758c4cd08e15c5c204adce3a372ff88067693`
- score hash: `sha256-633aa7766410da72c5bdc9731c453d9289686ed4edbbdd5ead6ec6c2917ed613`
- bundle hash: `sha256-2b93740e136cb3f5e9235a215ff24ab2442172286e8c32d10157c83aa3dc77e0`

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
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-9746b011921de143c84b7e5a3c7ddfdf2dd315bb3619b7e54606609377a13276 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-52b28cc5a0c68fff35c59b769c4aa7727dac55cb371894ce7101d9dd08e1c21b |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-bb6296cd54feef1e7ec84a236734e1da87e337dda357360904a9bd2a22674f8d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-7dfabb7c | sha256-0866abd7ad34e4af061417c9879af01fe8b79c40aee1d21835cdb1ecb0067b57 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-7dfabb7c | sha256-4bdab7537520ac8d6d86b28056c080b807b24e77ac16edfef5615d8cfe327bcc |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-4463fa3b | sha256-f554191e9b8989d144893f8ff1af38d83ba2afb09be4c8a5fd5980ea6b6e6362 |
