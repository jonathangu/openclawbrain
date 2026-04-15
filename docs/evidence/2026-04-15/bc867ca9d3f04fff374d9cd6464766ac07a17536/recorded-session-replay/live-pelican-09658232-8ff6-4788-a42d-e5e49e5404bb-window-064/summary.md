# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-064`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5f39bc7349702eda18b0d056342226b1aabb41caee42927ea480ba26a62daf2f`
- fixture hash: `sha256-841eefc5eecc02fe972ed7cac8e3716da5b289fe7edcf8c461503d651db37931`
- score hash: `sha256-4daef9907a3a082e3892d592f91b7dae5da85d45ac5bc5fbeec5c0e7babd53e9`
- bundle hash: `sha256-f6ca143550f4b262d07703f9e174ae0f29659cffdf6f2ad867cd0ae58cbc8926`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f5559e04ec3c75dd16ee057dcaef2391dd2363ce8cb9ccfcfa727aea97487dcd |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-56c62b253f36d46f4591e7f0ce40a45ebd10ea7483512c571880bcbb2396f90f |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d836a41d38e436bbb41d15c94a8178ed8a8274fc313422cf0faa7aeb560db751 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-61d21ff46e80a1ae54fe44ed7ceac56913e8f15689dae6238d5615193e60d5dc |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-6ab28b1e | sha256-b26a5d2c8af83ded3ac601ce76a4424d1f9956abb04424aea5494f825b4e0973 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-6ab28b1e | sha256-d00c7ac61ed8b379acc07c6c4629e986b4af60bb8f002e672936a358a470abde |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-6ab28b1e | sha256-b26a5d2c8af83ded3ac601ce76a4424d1f9956abb04424aea5494f825b4e0973 |
