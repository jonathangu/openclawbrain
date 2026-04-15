# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-173`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f6e44a71ffb544349fa10e1154a65bb6e77238a611db5acd86432535b5d68dc4`
- fixture hash: `sha256-3faebbeffb8f05bd64fe046d292ad1b3475373e375c449edb9cff67872d9f497`
- score hash: `sha256-2802adb5f2dceb36854ead407816e9d600847813d0b0bd76ef1c28a3448f1daf`
- bundle hash: `sha256-0e53fb93b343fa56a9de568fc38661f365fd466c4fef39767dfecd9b698ee683`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b418587cdea65dda940f9a601cf2fc169601499e945221393d659c55b40b8049 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-51e5c5fb21e2035267fbd71a496233a7443b46a4949ad5f2bef5cfffffb8e258 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-91a926cd04dede140a2beba59ff09f855740277cab1248de39629801f6d1c05e |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-b0b246c3cc792226b202d47a4b7816285d9a0427a32dd14c07c941283454cb1a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-1fa39b89 | sha256-f20ddf8334e458a44111844143b7d9b5f60eb6b0cbe387faf5144e3c0971dcd7 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-1fa39b89 | sha256-0773b975885f079a5b1d27cdf5047d56a323434edfd03b1e03cc9037569711e2 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-1fa39b89 | sha256-f20ddf8334e458a44111844143b7d9b5f60eb6b0cbe387faf5144e3c0971dcd7 |
