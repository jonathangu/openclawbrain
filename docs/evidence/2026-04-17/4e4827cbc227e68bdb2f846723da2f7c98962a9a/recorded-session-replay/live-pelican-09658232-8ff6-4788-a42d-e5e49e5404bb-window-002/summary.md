# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-002`
- winner mode: `vector_only`
- trace hash: `sha256-823695d70f7872b1ae9eafb6d1d27250c7a30f3c8da0fb3fac149eb03366ef43`
- fixture hash: `sha256-bd84df8e56b4c53a26fb492fdea7511a22aab4ac1b787c58633c40d2b1aa4455`
- score hash: `sha256-50904519461966dd3506c923259f67ad80bf7e4456e771b1bc382e214346c3f6`
- bundle hash: `sha256-3472b7b4be1537e38489914c5fda3af7ff3c346ac5d3a4ebf775379567b8cd15`

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
| vector_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-4ccf42a0a5609dcbac42225bbfa6a43d0ce50a817ff8379d53ed554c6eb37329 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-203d8195fd3f6fd4df670237026ab0c0e93e227262bc8238efc5bae95a90701b |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-313d6ebd86e938b7948be1f53cede8d0584cf686dfc9c41fc247a7280b504969 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | no | no | pack-0e6124dc | sha256-64b8c19bb1bd36a4d1ebea981b91d5bf8de874626ac797cfac48a7c62f111364 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-0e6124dc | sha256-d7a26791ba3e038456cc1a01442c3d53f48334508c6bc80c851fac7641dd0d9d |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-97d3ea15 | sha256-88d4d2e0fd22e5414c4338d432a7a0099b87351d2ac85f24391886acf8bea5da |
