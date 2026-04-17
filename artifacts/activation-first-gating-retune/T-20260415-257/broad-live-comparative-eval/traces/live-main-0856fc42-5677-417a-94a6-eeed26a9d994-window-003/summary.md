# Recorded Session Replay Proof Bundle

- trace id: `live-main-0856fc42-5677-417a-94a6-eeed26a9d994-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8112927457240059417bedc3d26ba052a003896d620c2316ad6b12373ef80eef`
- fixture hash: `sha256-14ad40161fa5c35ed07d9d394829c949bb081beaa26c47469b137af3b630df8b`
- score hash: `sha256-bbad194ad8d3d8b77756d1193a909d5574ed8871da9520d73cdec46a5608e9a0`
- bundle hash: `sha256-3f4491754f150e5e21581d3410d4091d510a533c872743c9c1912de373a6c663`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 9/12
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 1 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7e54eea5dd476d45e5e7ab52a9b0ed2c646fc990677d2858d9966f3baecd8936 |
| vector_only | 1 | 1 | 3/3 | 0 | 0 | 1 | 0 | 1 | sha256-5db8d98b1de5f013a1e4a43f6619087b171fed54efe4430bd51527acada61e0e |
| graph_prior_only | 1 | 1 | 3/3 | 0 | 0 | 1 | 0 | 1 | sha256-5ed39bce49b81bbeec5f84690333040779a313214084e36a325e877347060257 |
| learned_route | 1 | 1 | 3/3 | 1 | 0 | 1 | 0 | 2 | sha256-b8a7a63f07c16dd49a5a6469f3033f89b6ab78b763c51cc1c799ea0ca32408a6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 3/3 | no | no | pack-35378522 | sha256-30a7ad5cbf3c20746ffaaedd81d23f891a819abfbd918b8d66fe7c885866988b |
| graph_prior_only | turn-1 | 100 | yes | 3/3 | no | no | pack-35378522 | sha256-28a99ec60d7218447b27f261edb21c5bbdf62004efd8cffff813185bca8122ac |
| learned_route | turn-1 | 100 | yes | 3/3 | yes | no | pack-89bccde9 | sha256-b86a270126956e23870a54254e523188763ce84defea7dec060864cc258c0d54 |
