# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-203`
- winner mode: `graph_prior_only`
- trace hash: `sha256-304f24fec2ef73f307ee3b3bfeff3d4bd90894e7d9fa693794ad2f916befa2ce`
- fixture hash: `sha256-0d0f5f0a3dfc50799aa0a0583bb1e17204f3f01f50323b91030ad8276022d234`
- score hash: `sha256-49cc579e341ee9b2100ca913eb1c46e1920fff75316e985f255e5b8f2d4651f8`
- bundle hash: `sha256-1a622806aafaeb77b781d2978a3241c458f4182ddedb834be846a44b153c2f26`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c2db02be6aed4a54b1a82eb4486629fc6a8c812b69fc8bb1feef7e61858ba9b3 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-606a2316d45c2d796896a46b00a1db90e5344693aad4cec5a772257d3f5452eb |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-37ff7149c07043f3977154cdfea069cdbd8ec90845e1d10a3adf3c9630897fc0 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-43ea4d2abdd811b6f614489b24b20733e8036a5cb5959aaa99d2b884195bc6bd |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-886ff884 | sha256-d6a1539e104502c81aba697600e6a4143f1ae25872685c6e60d1aa4f3b3d3eeb |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-886ff884 | sha256-0d515c8633881481b2a966d4fbce8673947d2a3a0d2c32fc7bee7a41cdfaa3db |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-8441fbeb | sha256-f69e6c92ee62d4326852a8ea42f919116084306505dd667bf4659b6f301e3269 |
