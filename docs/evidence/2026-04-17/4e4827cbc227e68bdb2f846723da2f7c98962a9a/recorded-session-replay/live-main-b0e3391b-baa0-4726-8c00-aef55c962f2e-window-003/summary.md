# Recorded Session Replay Proof Bundle

- trace id: `live-main-b0e3391b-baa0-4726-8c00-aef55c962f2e-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1b2b13ce0910158e65496491abf6c903c5dfb4a0455709d2493e613749674539`
- fixture hash: `sha256-f054489b05d16d5a9f9a5c47426c143ae1eefae16f2ef4a677bba49745e4b5ab`
- score hash: `sha256-9cdedcf52710d96989cddfac5d175bae60135542bb72738377899c5e3e2f691b`
- bundle hash: `sha256-d73e1a95ce83d867482f38f6f5ebf2f6ea27bdfc4a666e0118f6eadc9b4fb4e1`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b85312fe12721c0ca336fefffefaf7611e66d0c3fa24585f0a8f1c80b737da2b |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a9cde1ae23b997d84ab8b74040075716ab6360984bcbbfb5d1535aeee329f2aa |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b27102652412808a66c0c9121758eaf5315d6d1a87c73f472057803e6c0bd28e |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-10da27b44e3524435530f93b17a549d408b594b3a058ab12a27833ff956c236b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-70d807d6 | sha256-48afc3d65a863f44a1d88e66fcecce047dd7f2a99763c033eebe7ff8824d6a2a |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-70d807d6 | sha256-4665da09d48238bbe5398e8a0ebe59f8c1f50698aa1b88d124295ecaf7e729aa |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-2989f463 | sha256-268dba7dcc9e2e12b1d009ea2232f2e37d3fd49ca6c6bfe3192888f137c2e1e1 |
