# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-4654a9d2-02db-4eaa-a316-86d131e91df8-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0107560e9fd434b7938c996a94e09516e9330df1381928365035d337054775c9`
- fixture hash: `sha256-ccf24038ed94c209310a49ea52fc2105449214d461e66d2dc1493bec54050346`
- score hash: `sha256-6f60e3d491619dd2b84b964698ba7f5e64bde4e46820934da7243d4a2dd70aa8`
- bundle hash: `sha256-02fa0694bc5412eaaa5e190bf06b6565ca48cb0cabec03cb3553b021015eb14b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-33cd39ad77bdbc78ea0e62a163e0f69b70fc53f35c07ff18076ffb99dd86c22c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9220e961da680c4fd2ea2980239eed3e19b718dd40a1cc5a08eabf156a64a63c |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ff3188112a9ede98c62cc6c220c5aa60488d2df71b1f9542c0a346e04ac6350d |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-20af6eab8dd2dea22b0ed12d91ffb5d0dfd64730ef03d5a685a2e57056d104a5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5a2dc133 | sha256-583b882669e2e80e7c26e401d84e9f75c0e68aa1700c692e923052c6a81f0065 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5a2dc133 | sha256-645d0ba61cb0368a7df3af8f1b6067614797de6bfd9851e66ab19710e23c70f4 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-9aebeec6 | sha256-7652bb94ed1ab728d287f3d41ee734de71fe94f300413893c2e86f571fbd712c |
