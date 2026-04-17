# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-197`
- winner mode: `graph_prior_only`
- trace hash: `sha256-dd29f792fbfbc606fc0ae81485babcea7a498bf8f85b66de2333918434925117`
- fixture hash: `sha256-a84a33537b8d24e458443c5c6b1cbd9d02b490a8b56c8f49f8509184e51ddc87`
- score hash: `sha256-3c2f306bfe305a137d66c1b91fbb2f5464c8d1941b52ebadc3fa015b82cd2eac`
- bundle hash: `sha256-b6ed9a6b2e4477396ddc5bd9f3dd504b5f86790b8dff0fe3d9a97af2fd9a5a18`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d6181c2140140f5786a710721c2d0cc92976577da480a328a542e8b790bc4990 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-83028bf61e31b6ea4a6c21f8f783fa5d8046a1355e9e69be4584b61f68b19263 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a4fae03745008fe3a651c82d5279f4184154feced2eb72be1658d2a29da02715 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-763a656a2a87beda4ef79c4fe9c20b42d666a0177af1fe5b3c61d97c7c3bbfbe |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e49405f0 | sha256-e1b99fdb554f715feee9da059ce82796666f57e56a729389e3d3fdf1b57506ec |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e49405f0 | sha256-bfe2cf2cee565074abf9f87b510c2b0867b748ca10ff8762559057dbf4ef3dff |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-e1935d1d | sha256-63572f933d65e94138f86d28c42065130da8d9f55cfd0f6b5210e2fd4143e9d5 |
