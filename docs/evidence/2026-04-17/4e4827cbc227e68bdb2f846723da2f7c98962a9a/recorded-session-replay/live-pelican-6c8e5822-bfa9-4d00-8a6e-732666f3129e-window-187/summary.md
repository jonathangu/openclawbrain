# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-187`
- winner mode: `graph_prior_only`
- trace hash: `sha256-eae72c8906ce053ade6bf66b6f03ddc87f48a19f8e1b50fd6f47ba9774ecb440`
- fixture hash: `sha256-80de414b90b70f70f1d2f2daf70e3430dc27d1af7b593fd0e1e1dfcb61676ead`
- score hash: `sha256-fb5df18303f199f35bd2bcd939aebcbc817d53aa29efc3cf575b9bd2c80136fb`
- bundle hash: `sha256-fcabf45ab305e66c85a11e61c9bd7484ed5353f26ddc124d56140a63d7e9f9c3`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0e2afd5c5e27e893dea21e62c6e8b163bef7241aac8748bba68a4d993b31b8a4 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d1599856172bf7bab5c5e13fb25cdbfb0c9e5058f41b47bb1eccf1e68cab60e6 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2e4790595613b491dc3a758c277e4e58aafe78a9de3b6108985bba8303ade8a4 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-6405ac66a6adb3c1f1ce30c79826a4a78b7fb289e849e9be8081ffb290b58434 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f67bf5b8 | sha256-d5893301e9820799446d5f528bbf16625696786c9ef073a8b42ed026cbbded85 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f67bf5b8 | sha256-3f7f0c08527f505e8f2aa4ee4132f66409e50ccb58f836d89ce331fe2ca58bcd |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-9fba3a59 | sha256-913d445c65d37804dab639e03a53ea2efed19370d8869895046db8ce1c556b2c |
