# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-002`
- winner mode: `vector_only`
- trace hash: `sha256-823695d70f7872b1ae9eafb6d1d27250c7a30f3c8da0fb3fac149eb03366ef43`
- fixture hash: `sha256-bd84df8e56b4c53a26fb492fdea7511a22aab4ac1b787c58633c40d2b1aa4455`
- score hash: `sha256-95cd17cdfe3239ca30ede77708fe75fc6b8e70c8bb58403124daf079d2a64132`
- bundle hash: `sha256-7d72b032412d6e478f82dce8254e65f0cef3f6dd60b079a35330b5b3db38231c`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | vector_only | 70 |
| 2 | graph_prior_only | 40 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 1/8
- phrase hit rate: 0.125

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-a1db4bbe90ab058f57bc7ae6a54f5aaf2daac0fc5ad242f5b0e6f3a965eb8e61 |
| vector_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-d98de325f13ae857184afdd6aa32080b643786318cc5923f79b5574377b1f24a |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-4443bbf2610e10073eacbb1779f76589bbd6155d6a100c01926da8a0ddcb1009 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-0ffc646ca90ed75166e87b1b3b7f07e8d8e3f91a0ac25790fc3c34f76742da13 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | no | no | pack-cf11cd32 | sha256-f71322ebc54ba084d988d41800e11116b87525ae5050691569aad9bc3db71d51 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-cf11cd32 | sha256-2b7805cba36776babea849c7b265a15313a2bac23c090fb84ef3772c067c899a |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-5884926b | sha256-f684397213bc7d18195e7743993532ae8644a67520d5ecce93136fc305b74c99 |
