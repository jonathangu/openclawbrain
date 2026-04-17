# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-169`
- winner mode: `graph_prior_only`
- trace hash: `sha256-fa0a3e5a2be78a517ccfe2e1e4b8f4e2529d6e5ac6ae4838bd2c1da5073ae788`
- fixture hash: `sha256-9c907c31d6df545ad3189fb66d2746fb0938842a92e6704858a51c0bdbc6d6a3`
- score hash: `sha256-2a47ab093c326feb127f5fb099ea801b879fbf88ad5930bc7802357bd63866a2`
- bundle hash: `sha256-fa5536594824ff2952854ccba9aafcf8be1bac35b89db012509aeeccef082d38`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-984a47e062d84a4c2db4727f0e783a355cfd91d65c98b2c3a27a24fa9103cec7 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ac8307154363196c418d00dd49a3dc27bb73faf8e76ede630b22f45b2102690e |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0d325f38dcbf48642b74154a70fd90c57921dfcb39370024329bd56fb3dffc59 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-1d3ba4c45b109886fe75055e60ce14699620d856aaa32ac70f88a60c8facbb31 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-922e2291 | sha256-1d58c81367b9fc352aea6279d62a8e3d31f7d1d92e98d7650de1946d1983cade |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-922e2291 | sha256-1dc9deda621f9f5d3db3e7ddee52192dd936e74bdbd5d8ceb627030790357074 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-84ec5ffe | sha256-be2ef5f82ad5a1c3846fbd819840f4b780ddf4d5179805b39f8d37eba56e6895 |
