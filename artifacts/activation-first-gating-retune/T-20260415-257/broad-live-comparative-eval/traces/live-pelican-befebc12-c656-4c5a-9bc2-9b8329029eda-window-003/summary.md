# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2b0cf5e7b3f41bdf7d892185413608401c24e9c3ad252c16335ba4fe2f91cdd3`
- fixture hash: `sha256-ed8248c9b476e9fb2d02b9891cc8e11da35a8ba49c308ca9793fd2e0cd5daeaa`
- score hash: `sha256-e4123fb580c4319fc60a06d71c8762b5fcbf1e4d391acd4fb20ec2884229ca86`
- bundle hash: `sha256-2528615c76fa34bf62819134612654c63188cb60d9241433791a2d647c746a34`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-368ba7ad0e0062707beb6bc226c2cae8531ed592ec4225d05a99c6ab4df81531 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-7c0b1baa40fb8712454189b3359fde8c742769ddb02c6ee5984925671aaaba7c |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-007243bc46347d340a3a23c7b1c33987bb09ce2f578bba4480e6796401484403 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-0cabacf9029e74b3b2cbda5a6433db30894119e047cb3fb9bf96b3bddfd3f445 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-bc49e4ab | sha256-4466328bb6008684a26826d02787d4b911587b96963e7060b1d50ff8f91725ac |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-bc49e4ab | sha256-d64b81ee3bbc07204a3a76794d244464812fb8a4bb36702e6c571d0cdf4089e4 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-3a0559d0 | sha256-90b5d4a2d978312592e155005f89f62b8168a49eb08b122e8248833cacfa2d06 |
