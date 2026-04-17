# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-016`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6bbdcea0bcebe80c4396f90ecefc21d80712c695057e642f25e10243f87c4c7f`
- fixture hash: `sha256-e1e5a273f33109e97564303fef433c5ce8b0488cb943d73553c438e3bc4b82f9`
- score hash: `sha256-fe72dcd55ea71319f8df3076d8a03026f0223b40458f9ac05f7daa67bfb8df28`
- bundle hash: `sha256-ca9f4bc54526bbce58efd437555848645afb668942fcc6537f4f82f77de322e7`

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
- phrase hits: 0/12
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d7d1f1cbb97b76e814b60c9940bdeb937cd1496e6d58b8c2941e362dd90a4031 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5373881b7d756523a1c6669b6d61a60491930380ccae4a122f9037c132eaf443 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1f140c42e343804efe19c57492f92a5faf59f9fcccff06cd7a3ec0a20eb4e22f |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-b66aaf57f2948734b9bcc6be046dbbe9f897cde04ca93cfe370fb97b04033ca8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5a0abaa4 | sha256-9aaf88e733a7213f34b70f94755f8d35a5711e6566d1b0b40e275b964b71c2db |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5a0abaa4 | sha256-44f6929ef08e67bd1251e96df8d98d2ba2cbc07bcb1d2cd7ff5bfbf4364fb5db |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-2474eb81 | sha256-f0a249085826c0c50ca5029da75bb71e6925a7ef476b336a674f9140833bad8a |
