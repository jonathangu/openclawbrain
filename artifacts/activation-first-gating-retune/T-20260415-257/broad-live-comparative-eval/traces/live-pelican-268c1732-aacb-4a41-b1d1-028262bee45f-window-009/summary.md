# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0c4b362666455e64ccec1b7026696e6d7ee86b07af9d91203554a5f880643a7f`
- fixture hash: `sha256-ed322207ac696cb8afb94d5f75c53ba4423b96ed55d3a35abcef96ba37d6147e`
- score hash: `sha256-74ce6aa06a135522021f09e9bdc7b38350087b24e3160d4b9d29ba2a8879131a`
- bundle hash: `sha256-1cb77b89bb1f81a115c21042890d109a54443ff39499e7886bad98cc042d3248`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-659f0a773cf1e71603ffb20a5a28aebea6d5db6139dfdf0255a605e7868cd22c |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0f6a182b8b98361678d3697da808f7b372ef98a47c6f291cde03f782e5af81ab |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7201a8ed544389705a11ba93b8f1dedb90685fb024048c7ba2706847e1c70f25 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-5851f6255a5750c7f0ae23f7ee15457afa6ac2975f28bb601ac6783d3ff07af7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-402221a7 | sha256-6e6bc3281eee693756ce4aea76c38e6d0897fff9f8e24f2cfc0f6e7e792b4b64 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-402221a7 | sha256-85f8b4d947645960eeea146718f966f45bf04952ec8e2b9901c60dea56665f6d |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-0e6fea5e | sha256-496e8d105798cdda6705ca57b06969a002bedca99d8d0dc416a7160aa3a16458 |
