# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-011`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c761e47d47331bf575b7d002c131402195e8ac49d688ce355a015b14825d3acc`
- fixture hash: `sha256-128e53eb7404fc5c5e08cc33f7657166db8766e76b0fe254b4c32e80c9220dde`
- score hash: `sha256-d4721d163d4914ab1f587a1e625711fdea04510c38a67cd69bae8e0fac584fa6`
- bundle hash: `sha256-e273c83397b5f262aae831a4e9ffc770258c52d4dc9238f64f1010f922be59e8`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b1631644a30db1cf3d02f7c72d9e973f8085c5ed6318e74e1e83701e3e901455 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-82a466057cc156d5f04fe09dea951ac2677e07dbb760a68da9e6ae715063994b |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b570459aa69c723d77d4f3196113fa2647ee6320efb55a0a3c12c7d6d40cc545 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-d726464e6894cf316cf32cd17b85393b153b7f487ce6817c7f90713ae487ae54 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ba453d4e | sha256-2c2205f114348de5ad2eb88d96db6aefa03832aabe38148dd3f3a76e7ab348b5 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ba453d4e | sha256-e61666eb6aee4aa7be1b53cd43081be7cb4f1165234431c2aeed64df6f96b275 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-590c7641 | sha256-95d4e526322da93a788f26852e48bbc94f9cafe27d13d731af379f3fd2c95c6e |
