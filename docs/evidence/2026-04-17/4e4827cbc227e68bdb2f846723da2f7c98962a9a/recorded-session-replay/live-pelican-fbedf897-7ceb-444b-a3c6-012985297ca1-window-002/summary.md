# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-fbedf897-7ceb-444b-a3c6-012985297ca1-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4e6bcca0889e112786ccd30d6dc08d693afeab2955ee3e21db9f09dfe3094e0f`
- fixture hash: `sha256-14ff100a8ccae36fc1c57494dcba2b6e1338cfd708e5c890121212b4f7b539d1`
- score hash: `sha256-0b5e12b07030df79e800820c755d7b2aff8445c94d4d4a8b75e8ad2e1536362f`
- bundle hash: `sha256-c32c14663746873edcec703653378a2f5756455f26b7ec0070de02ab60825425`

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
- phrase hits: 0/4
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-86a96bb1f89c4f625498603269cc86fe2157c50e9372e11582b94c39873a6510 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7d80604a6e0873bfc60c180bc433b9370a52fcb26afe3880c6da5b98c24d2433 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a7f5c3b25532c987d41000de27b9c2fa7aabe7584a58bc02220332ecefb826c1 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-bbd32402ed2c56a6e8a379b8c8704b10510ee65f0513c46cad6d501f50a48db6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-2c9a1e97 | sha256-a2270257e2888c94929b3ca329bffe77b5d817c4ae6ccda2448c43891edd456d |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-2c9a1e97 | sha256-de1f0f76d85476b90dc3ece711bcf31ae222109264522180e85d2880b807f04d |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-b8c6740c | sha256-3db19dfcc13a00009a79885b696e635c7f9b234325a97c19c0f1cd265831f31d |
