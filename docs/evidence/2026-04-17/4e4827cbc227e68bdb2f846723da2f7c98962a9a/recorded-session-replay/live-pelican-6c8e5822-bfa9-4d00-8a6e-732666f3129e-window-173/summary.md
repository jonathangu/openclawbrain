# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-173`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f6e44a71ffb544349fa10e1154a65bb6e77238a611db5acd86432535b5d68dc4`
- fixture hash: `sha256-3faebbeffb8f05bd64fe046d292ad1b3475373e375c449edb9cff67872d9f497`
- score hash: `sha256-23f3b4b8ebd362c739b508079d5c045a68e90641c996097e676ba9fa8ad33508`
- bundle hash: `sha256-21e80fcfeeac14eedb37422d72d8359ff924aac6938da239705c30ac5c414296`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b418587cdea65dda940f9a601cf2fc169601499e945221393d659c55b40b8049 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-cfb2ef1c6e5b5c81948dbccac332155c18171813ba2b8e762b88a8fe82e9339a |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-58ce5c461bb10a0a959687e85aad3ffd49380d0bda1441c0fda91cbd469a1f02 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-fa798b58aa73cbf374f8127f81f40ee8623adf59d348b2cde8e1c6bbf18ce200 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-64df2668 | sha256-195e2e9260d7706e84b821f64e0b736e6baa82e926755208b77e6f0242f0de9f |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-64df2668 | sha256-6a85e63aab3c4c0a6d5f49a5b47e0ca0d873378c713b1e417882c642d7f76446 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-88e28f3f | sha256-5fd09e1eb78c4055e9598f8de0a935ccb157515f88ad2639c6e1b3bbfc8acc7b |
