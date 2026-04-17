# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-173`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f6e44a71ffb544349fa10e1154a65bb6e77238a611db5acd86432535b5d68dc4`
- fixture hash: `sha256-3faebbeffb8f05bd64fe046d292ad1b3475373e375c449edb9cff67872d9f497`
- score hash: `sha256-bd5735247319e5e2f2a907fa867f4d8a4cdc1116cf557955fad7a58327012c20`
- bundle hash: `sha256-70e5798c39f4c67fd289f470c7172add217a844209c6e74ccf9768eceea2987f`

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
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-838e19c752171254e1ceea86bc141d80a59b1c761326a005a052daa61d1792fa |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-4f0496dbffa48694bbfdd9009af7192b3a9f3447b7c7646fecc3477ab19e111e |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-b9d0a8ab6b0d5e423a76fa6090617ed5b3378143e8636f3245c69d64c194489c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-3cc67451 | sha256-8f3aea1b2b94393831d30feaa87774afa65cc1fe447869e6572015d31e8cca52 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-3cc67451 | sha256-725e927779206736be624d817a38b2017f9b1663f6b3b56df1901fcfa1decfee |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-60c9dd28 | sha256-f964f40350832fa937dc099ac3fb46d68b7ebd11ab3c9f9ec64b7606a2224256 |
