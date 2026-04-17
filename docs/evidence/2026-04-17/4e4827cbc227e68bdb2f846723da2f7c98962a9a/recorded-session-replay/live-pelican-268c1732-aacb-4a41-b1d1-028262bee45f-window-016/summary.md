# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-016`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6bbdcea0bcebe80c4396f90ecefc21d80712c695057e642f25e10243f87c4c7f`
- fixture hash: `sha256-e1e5a273f33109e97564303fef433c5ce8b0488cb943d73553c438e3bc4b82f9`
- score hash: `sha256-d0956dae3bed312027c16795451fdd05c99f17412d6dd40c56bd7ec408785be2`
- bundle hash: `sha256-350abaa2a17c480adbafb245b8bf2969cae2e4346fa65e4640589c9f6ba5fc9a`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d7d1f1cbb97b76e814b60c9940bdeb937cd1496e6d58b8c2941e362dd90a4031 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1bc958739f59393fc57c3cf463f186b5f62c5c0e399bba44b33ec954ede0ae99 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bdabb17ce081f24f603a9dd7b1036b505f030b39df4ffd5d5a5accef0977829b |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-d810aa61249cab85a339a6fe56335402a9764069f456f88b33512de1e0f41734 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7a0a1f42 | sha256-b386d57de037c1d8c8702f3eb1208f2f664ede754878795b5f69a218aefe178b |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7a0a1f42 | sha256-6ac5999e2838c56197f1aff2e22b60654b1c23aa48831c4b54794f7c8d6367da |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-4474501f | sha256-20c4505ce917ca7c2c0704560213876201db07b23fc0219eeee0a3c131b8eb96 |
