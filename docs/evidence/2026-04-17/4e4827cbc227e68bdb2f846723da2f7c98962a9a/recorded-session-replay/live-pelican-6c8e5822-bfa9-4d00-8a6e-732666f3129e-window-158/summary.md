# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-158`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1daacbdd680cf4033ed5d9fa2efa105e6544ffe1129c7600ff85b76c0c2f8393`
- fixture hash: `sha256-ef7e749ef838de36d236aa29e0590a88a86c6b42be12cb84bf00123ad9c263a6`
- score hash: `sha256-7dcc0008ee484117be0b9aa91baac370996330edbc28b6ae4221ae01c7f25cc1`
- bundle hash: `sha256-8480fa8d871ca7d5a9fd32bf9f0d46653ed2681b81ef1aba6a5740fdccdd7a12`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-715391ab03706266e4dd92a9d6ff099345f003fb7379bf779cf731b9d18a7950 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-92513eb1dd512931d83a5ba6b861b1f0a20a6ebeb7b66063c0f578488c4ff35b |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ec0bf94992ba5afed6bdfe5b510e805ef94896c0f42a0f5984791f972c6ac131 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-b206546638c2711ba777e360ede4ee446010d931ad564735b5696b1061ef81d8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a937f461 | sha256-51513af1907ac950b2425c33ee8f4b4c7249860c5266ebabb3322fab3366504a |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a937f461 | sha256-e84688db61d06ab2f3982b720428704b1836a39675fd092205b1d60e0b4930a3 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-ca93fc84 | sha256-66678e106e42b3925253514acc6c6d6bd9169b87f2c070bf3c969afdb828b558 |
