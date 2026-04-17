# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-016`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8d30d0b2ffefbdcd1e1a89d75d980761c51cd05c50f2c3cf1f693944186af036`
- fixture hash: `sha256-029c6b1d164f9bd1c4692f0184b6bb3b57e3ba2e59663e9c61a6962698d01e73`
- score hash: `sha256-a8e7143c2263d75e12b44e156342297e3a233d4b2ac3c6914dacb3a2aacce025`
- bundle hash: `sha256-625995a436c08de503c7b0112b1c4d149246d287748e472641efb147be137781`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-6eb030a8259079868b419f4ae1a6c389dd22240eac5e867e187ea0fab1adf6c7 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d22cdf865b44b671372c506e09d9751b7209ab01c24b422ed62f8204c9f29fd7 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-5ce065bf98b8a33e6859679154f3652bab6c7bb3d70e94472db0e2f68dcbe695 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-c4c988b241777bcfa9964c5e79d10eac734bac0ebb379b7b6f508af3405badcf |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-778f6206 | sha256-79b4923412613bace62d8eb5550ac293cd8a2460ca45221fc1d25b9382894d73 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-778f6206 | sha256-bf4084862d79e207b2258504fe6c9c2d7987284970d689989917754baaad2685 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-ca770d43 | sha256-364a929fbea6bbe09013c2a389677a760c9cf34b36a1b52c588c39d8176234e5 |
