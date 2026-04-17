# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-184`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d479c91d117044f49ae49da499694fdbe9a9bce3b101e2f906d0092b46536940`
- fixture hash: `sha256-afb53fed27fe0fd6a6ad4e067cb4e140573e8cbd954bfddd658b0c3c6c424a0e`
- score hash: `sha256-17f79bd34f3dbd2f56e37298df8e9e486e8520ec64dc140c3437a78400eadb7e`
- bundle hash: `sha256-08edc13143d3073f7a6b73015c58696fe61752df2fa0f6506a62033f279d823c`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 80 |
| 2 | learned_route | 80 |
| 3 | vector_only | 80 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 6/12
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.666667 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.666667 | 0 | 1 |
| learned_route | 1 | 1 | 0.666667 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c2f4b660a3e5d5f4a920994b92d0eae72726c74b613f3139fcacbac22692626d |
| vector_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-c3f326225cbbdcf616b3d705501f57cb6c8225ddd8d65f5ebc01a87f4b548e81 |
| graph_prior_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-d8678d2b40a7b61e16983eff7535644ffc46d163d4e67663e32de0c52ece2c79 |
| learned_route | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 2 | sha256-2a2a11fe42d7f5342bd100bb28d2edd18ee6ac3af1fb3119d8b4db586472aceb |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | no | no | pack-3c2a1aba | sha256-787c1832d07ce9ebb2a73c479e89e16cee43eb96761f3e22542393fef9d4fc82 |
| graph_prior_only | turn-1 | 80 | yes | 2/3 | no | no | pack-3c2a1aba | sha256-ed1ac987fdcf7d5570916781d83dba19733c444d40eb63b08d9cf30ddd2ae37c |
| learned_route | turn-1 | 80 | yes | 2/3 | no | no | pack-81d4effd | sha256-34b36b85e74a7e025b4fd0f7d27f9b68df60e907485deda342138c14aedb2914 |
