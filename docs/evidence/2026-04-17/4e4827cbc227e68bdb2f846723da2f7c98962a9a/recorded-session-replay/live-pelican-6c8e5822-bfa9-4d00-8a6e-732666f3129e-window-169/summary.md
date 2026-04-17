# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-169`
- winner mode: `graph_prior_only`
- trace hash: `sha256-fa0a3e5a2be78a517ccfe2e1e4b8f4e2529d6e5ac6ae4838bd2c1da5073ae788`
- fixture hash: `sha256-9c907c31d6df545ad3189fb66d2746fb0938842a92e6704858a51c0bdbc6d6a3`
- score hash: `sha256-92d66557a2b2f0a22ec7a5abc2d6b77f21acfcdfd264ccee55189fa3de8a34ae`
- bundle hash: `sha256-0e5b53d4b44058d0c79122dbb6b17ac71d9e48eb19677eaddaafdd3b65a5318e`

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
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4a6a78434f25913bfe7cfdd53b805a3fabd7dced8a7584f50cb6682e01bfd419 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f20e22bbf7487928d301d72448afffe3e4e906de06197fa8953de2437d950b68 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-402b9d0d9cce7879260733fe86a4746768628e96b3dc39ba40fdfc52b8238ef7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ba46d4a8 | sha256-31e3781f16d9621dfb375fcbb3ce3937998abeee5332734df6c07893ce07f8ab |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ba46d4a8 | sha256-6caf2215b25b9c798347c1aaea57a93fc364acf953aea01a70ab970615251972 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-ad051215 | sha256-d4696e86be0649e8c42aed0afe59898da7bc86dd7ef8a2be056a6068e38c0320 |
