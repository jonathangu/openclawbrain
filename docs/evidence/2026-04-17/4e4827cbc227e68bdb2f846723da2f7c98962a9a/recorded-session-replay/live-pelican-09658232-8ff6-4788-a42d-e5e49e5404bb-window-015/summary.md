# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-015`
- winner mode: `learned_route`
- trace hash: `sha256-6e0ff46b57f7c50af20d23a4d8a3f648535a36cc4021c3842ecad10617366b5b`
- fixture hash: `sha256-d2c3dec4ca105c441633ffddbfc56cdd05e4790ddeac1ae7cb9c9e93a7fe734a`
- score hash: `sha256-192ff65c464874879fc77543e06613a87a9636566541d26882068de15228aef3`
- bundle hash: `sha256-108afb6329691ffcc281671f6019db5f3ea6cd20a80b8f7ca0ec97ea724e45e7`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | learned_route | 100 |
| 2 | vector_only | 100 |
| 3 | graph_prior_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/4
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 1 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-48620826d5383928480bcd6e76b64505c9f9a20a21654ee2da4ad581ffb660b0 |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-7b8d1467915eea56c230b6dd7d96348ee1445cfd022c2f671080d8ae644baf97 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5fa756bdeab33aaaba722f4af2e6153a0e25223b649ee23cb79a290b2002a3fa |
| learned_route | 1 | 1 | 1/1 | 1 | 0 | 1 | 0 | 2 | sha256-17c59d0ee217a14d782c56e842f55b6c242b000be64ebb0ab755f623ab4e1d87 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-aad2492d | sha256-cce1e047f75e90b395cdba0b5d608ae15235f091b20316e6c911a4da36e95bd3 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-aad2492d | sha256-505cb89ad1679c085f6b2c3a029e7353e556bc10f5abbaa18929a105f5d8f56a |
| learned_route | turn-1 | 100 | yes | 1/1 | yes | no | pack-08e8d78c | sha256-6689cdcfc1a8d744907631a4f704a3fc34fb5d0b07822ef75d62a133b6e5fe32 |
