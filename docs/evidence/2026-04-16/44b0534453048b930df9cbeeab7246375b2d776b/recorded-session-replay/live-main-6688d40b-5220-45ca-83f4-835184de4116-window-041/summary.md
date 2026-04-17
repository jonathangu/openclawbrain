# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-041`
- winner mode: `graph_prior_only`
- trace hash: `sha256-cd17705850f5fd87f770e4757922f483be90c3dcc5bfff44d696c49e62560cb7`
- fixture hash: `sha256-743937076adce554085fa9dd3236567f573df76180477a11d06a07f43c4044bc`
- score hash: `sha256-5f5b05710c9b59b5db6cb3d170cf1620842ac65be6ee459d718fd2c7994e68d5`
- bundle hash: `sha256-194214fbb860bb0c0a269ccea1a370cffafc73edafbe1116f5e840aa7a64b521`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-38ffbd4329a21a765f40f1a44ad7d1cc0603504c91e4e697e7b573151d0b2478 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e2d0455992029fff0d3f07af6ca204c4320bee53ade0859912c450ba445836cd |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2e68ad32a343f87f12f565f278504fa7ef873a6292e665b0a7f29650067e80e6 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-036cf3081c2d5fe872755cba5cb498e6bd6f9fd7645338a65b928879f66de7aa |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-bbd0496d | sha256-8dce9e4619cf77b4f0aeed2c9a637fe586ac21c15385afe6a730713793d97bfa |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-bbd0496d | sha256-10b8b57e3dc077b46a2ef1ccd46cf6051911d86d21e564aee9a5bffea06331e8 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-fb84d646 | sha256-5ab3df15e09de82fca35000923a2e1c4614c20e52bd9571d7dcc3cc3213c6024 |
