# Recorded Session Replay Proof Bundle

- trace id: `live-main-b0e3391b-baa0-4726-8c00-aef55c962f2e-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1b2b13ce0910158e65496491abf6c903c5dfb4a0455709d2493e613749674539`
- fixture hash: `sha256-f054489b05d16d5a9f9a5c47426c143ae1eefae16f2ef4a677bba49745e4b5ab`
- score hash: `sha256-cb94e3c512111b94323150aeb2a7931233728730418643542da751b81ed654f1`
- bundle hash: `sha256-94d18ed6cc4d8b230a166b79c81a204038922b99ce3e1b6117e78468e1f0da0d`

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
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4aefeefa8f74c97030be9cda13d9745a25437e8758766a6f6ce7693a57c6ad21 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5a8b4033b7778a6fc36d54daae1ae4c1037f768ade20e6ee0580b2c5ee3be2ce |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-4386fb562a33fa03ffe3d2d271dc3564b133fb680f4260580430ac7a869e28cd |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0805fc3c | sha256-bc6445894691dbcb9c66c68666931605162178c59ece833fac9328d36ab2e43c |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0805fc3c | sha256-32729fdc6e57d9b2890764c7f6c53dfbc320203e80a5e35dafa8d40e19246acb |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-c0b7e8c9 | sha256-a86c488be46a677e292f79d2fd0a750e66a407a4ca567004d7b30346ceae8620 |
